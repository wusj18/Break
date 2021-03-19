#!/usr/bin/env python

import argparse
import glob
import logging
import os
import re
import sys
import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from evaluation.decomposition import Decomposition
from evaluation.graph_matcher import GraphMatchScorer, get_ged_plus_scores
from evaluation.sari_hook import get_sari
from evaluation.sequence_matcher import SequenceMatchScorer
from callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
from transformers_local.models.bart.tokenization_bart import BartTokenizer
from transformers_local.tokenization_utils import PreTrainedTokenizer
from transformers import MBartTokenizer, T5ForConditionalGeneration
sys.path.append("..")
from transformers_local.models.bart.modeling_bart import shift_tokens_right
from transformers_local.models.bart.modeling_bart import BartForConditionalGeneration, BartForSequenceClassification, PretrainedBartModel
from utils import (
    ROUGE_KEYS,
    LegacySeq2SeqDataset,
    Seq2SeqDataset,
    assert_all_frozen,
    calculate_bleu,
    calculate_rouge,
    check_output_dir,
    flatten_list,
    freeze_embeds,
    freeze_params,
    get_git_info,
    label_smoothed_nll_loss,
    lmap,
    pickle_save,
    save_git_info,
    save_json,
    use_task_specific_params,
)


# need the parent dir module
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))
from lightning_base import BaseTransformer, add_generic_args, generic_train  # noqa

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


class SummarizationModule(BaseTransformer):
    mode = "summarization"
    # loss_names = ["loss", "generate_loss", "copy_loss", "copy_or_generate_loss"]
    loss_names = ["loss", "generate_loss", "cls_loss"]
    metric_names = ROUGE_KEYS
    default_val_metric = "rouge2"
    tensorboard_writer = None

    def __init__(self, hparams, **kwargs):
        if hparams.sortish_sampler and hparams.gpus > 1:
            hparams.replace_sampler_ddp = False
        elif hparams.max_tokens_per_batch is not None:
            if hparams.gpus > 1:
                raise NotImplementedError("Dynamic Batch size does not work for multi-gpu training")
            if hparams.sortish_sampler:
                raise ValueError("--sortish_sampler and --max_tokens_per_batch may not be used simultaneously")

        tokenizer = BartTokenizer(vocab_file="tokenizer/vocab.json", merges_file="tokenizer/merges.txt", model_max_length=120)
        model = BartForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        
        super().__init__(hparams, num_labels=None, mode=self.mode, model=model, tokenizer=tokenizer, **kwargs)
        
        # self.model = PretrainedBartModel.from_pretrained('bart_base')
        # self.cls_model = BartForSequenceClassification.from_pretrained('bart_base')
        use_task_specific_params(self.model, "summarization")
        # save_git_info(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)
        self.model_type = self.config.model_type
        self.vocab_size = self.config.tgt_vocab_size if self.model_type == "fsmt" else self.config.vocab_size

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=self.model.config.prefix or "",
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"
        if self.hparams.freeze_embeds:
            freeze_embeds(self.model)
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        # self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.decoder_start_token_id = None  # default to config
        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id
        self.dataset_class = (
            Seq2SeqDataset if hasattr(self.tokenizer, "prepare_seq2seq_batch") else LegacySeq2SeqDataset
        )
        self.already_saved_batch = False
        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
        if self.hparams.eval_max_gen_length is not None:
            self.eval_max_length = self.hparams.eval_max_gen_length
        else:
            self.eval_max_length = self.model.config.max_length
        self.val_metric = self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric

    def save_readable_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """A debugging utility"""
        readable_batch = {
            k: self.tokenizer.batch_decode(v.tolist()) if "mask" not in k else v.shape for k, v in batch.items() if not isinstance(v, list) and k != "copy_labels"
        }
        save_json(readable_batch, Path(self.output_dir) / "text_batch.json")
        save_json({k: v.tolist() for k, v in batch.items() if not isinstance(v, list)}, Path(self.output_dir) / "tok_batch.json")

        self.already_saved_batch = True
        return readable_batch

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        # copy_tgt_ids, generate_tgt_ids = batch["copy_labels"], batch["generate_labels"]
        generate_tgt_ids = batch["generate_labels"]
        # copy_loss_mask, generate_loss_mask = batch["copy_loss_mask"], batch["generate_loss_mask"]
        # print("batch_target_id:")
        # tgt_id_array = tgt_ids.cpu().numpy()
        # print(np.max(tgt_id_array), np.min(tgt_id_array))
        # dynamic_vocab = batch["words"]
        sub_ques_num = batch["sub_ques"]
        # dynamic_vocab = torch.split(dynamic_vocab, 1, dim=0)
        # dynamic_vocab_mask_list = []
        # for vocab in dynamic_vocab:
        #     # vocab = vocab.unsqueeze(1)
        #     one_hot_dynamic_vocab = torch.zeros(1, self.vocab_size).cuda()
        #     one_hot_dynamic_vocab = one_hot_dynamic_vocab.scatter_(1, vocab, 1)
        #     # vocab_mask = torch.sum(one_hot_dynamic_vocab, 1)
        #     dynamic_vocab_mask_list.append(one_hot_dynamic_vocab)
        # dynamic_vocab_mask = torch.cat((dynamic_vocab_mask_list), 0)

        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(generate_tgt_ids)
        else:
            decoder_input_ids = shift_tokens_right(generate_tgt_ids, pad_token_id)
        if not self.already_saved_batch:  # This would be slightly better if it only happened on rank zero
            batch["decoder_input_ids"] = decoder_input_ids
            self.save_readable_batch(batch)

        def copy_loss(copy_logits, copy_label, copy_loss_mask):
            copy_logits = - torch.log(copy_logits + torch.tensor(0.0000000000001))
            copy_logits = torch.where(torch.isinf(copy_logits), torch.full_like(copy_logits, 0.00), copy_logits)
            mul_res = torch.mul(copy_logits, copy_label)
            mul_res = torch.sum(mul_res, -1)
            mask_loss = torch.mul(mul_res, copy_loss_mask)
            loss = torch.mean(mask_loss)
            return loss

        outputs = self(src_ids, attention_mask=src_mask, dynamic_vocab_mask=src_ids, words_attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False, output_attentions=True)
        lm_logits = outputs[0]
        cls_logits = outputs["cls_logits"]
        # copy_logits = outputs["copy_logits"]
        # copy_or_generate_logits = outputs["copy_or_generate_logits"]

        crossentropyloss = nn.CrossEntropyLoss()
        # smoothl1loss = nn.SmoothL1Loss()
        # print("cls_logits.shape:")
        # print(cls_logits.shape)
        cls_loss = crossentropyloss(cls_logits, sub_ques_num)
        # cls_loss = smoothl1loss(cls_logits, sub_ques_num.float() / 15)
        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_barsides ignoring pad_token_id
            # ce_loss_fct = torch.nn.t.py, beCrossEntropyLoss(ignore_index=pad_token_id, reduce=False)
            crossentropyloss = nn.CrossEntropyLoss(ignore_index=pad_token_id)
            # assert lm_logits.shape[-1] == self.vocab_size
            # print("lm_logits.shape:")
            # print(lm_logits.shape)
            # print(tgt_ids.shape)
            # tgt_id_array = tgt_ids.cpu().numpy()
            # print(np.max(tgt_id_array), np.min(tgt_id_array))
            # print("generate info:")
            # print(lm_logits.shape, generate_tgt_ids.shape)
            generate_loss = crossentropyloss(lm_logits.view(-1, lm_logits.shape[-1]), generate_tgt_ids.view(-1))
            copy_or_generate_loss = 0
            copy_loss = 0
            # copy_loss = ce_loss_fct(copy_logits.view(-1, copy_logits.shape[-1]), copy_tgt_ids.view(-1))

            # generate_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), generate_tgt_ids.view(-1))
            # generate_loss = torch.mul(generate_loss, generate_loss_mask.view(-1))
            # generate_loss = torch.mean(generate_loss)
            # copy_loss = copy_loss(copy_logits, copy_tgt_ids, copy_loss_mask)
            # copy_or_generate_loss = crossentropyloss(copy_or_generate_logits.view(-1, copy_or_generate_logits.shape[-1]), copy_loss_mask.long().view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id)

        # coverage loss
        # c = torch.sum(lm_logits[:, :-1, :], dim=1, keepdim=False)
        # a = lm_logits[:, -1, :]
        # coverage_loss = torch.min(c, a)
        # total_coverage_loss = torch.sum(coverage_loss, dim=[0, 1])

        return (generate_loss, generate_loss, cls_loss, copy_loss, copy_or_generate_loss)
 
    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["generate_labels"].ne(self.pad).sum()
        logs["bs"] = batch["input_ids"].shape[0]
        logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
        logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()
        # TODO(SS): make a wandb summary metric for this
        return {"loss": loss_tensors[0], "log": logs, "generate_loss": loss_tensors[1], "cls_loss": loss_tensors[2], "copy_loss": loss_tensors[3], "copy_or_generate_loss": loss_tensors[4]}

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        generative_metrics = {
            k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
        }
        metric_val = (
            generative_metrics[self.val_metric] if self.val_metric in generative_metrics else losses[self.val_metric]
        )
        metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
        generative_metrics.update({k: v.item() for k, v in losses.items()})
        losses.update(generative_metrics)
        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["step_count"] = self.step_count
        self.metrics[prefix].append(all_metrics)  # callback writes this to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])
        if prefix == "test":
            print(len(preds))
            print("*" * 100)
            name = ["decomposition"]
            csv_pd = pd.DataFrame(columns=name, data=preds)
            csv_pd.to_csv("test_preds.csv")
        return {
            "log": all_metrics,
            "preds": preds,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": metric_tensor,
        }

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)

    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()
        # parser.add_argument('--eval_max_gen_length', type=int, default=None, help='never generate more than n tokens')
        # print("_"*100)
        # print("generate:")
        # print(batch["input_ids"].shape)
        # print(batch["attention_mask"].shape)
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            dynamic_vocab_mask=batch["input_ids"],
            words_attention_mask=batch["attention_mask"],
            the_index_mask=batch["input_ids"],
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            num_beams=self.eval_beams,
            max_length=self.eval_max_length,
            output_attentions=True
        )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        source: List[str] = self.ids_to_clean_text(batch["input_ids"])
        target: List[str] = self.ids_to_clean_text(batch["generate_labels"])
        preds: List[str] = self.ids_to_clean_text(generated_ids)

        # preds = []
        # target = []
        # for word, tgt, generated_id in zip(batch["words"], batch["labels"], generated_ids):
        #     word = word.cpu().numpy()
        #     tar = self.ids_to_clean_text([[word[x] for x in tgt]])
        #     target += tar
        #     pred = self.ids_to_clean_text([[word[id] for id in generated_id]])
        #     preds += pred

        # ori_encoder, ori_decoder = self.tokenizer.encoder, self.tokenizer.decoder
        # preds = []
        # targets = []
        # for generated_id, tgt_id, encoder, decoder in zip(generated_ids, batch["labels"], batch["encoders"], batch["decoders"]):
        #     self.tokenizer.encoder = encoder
        #     self.tokenizer.decoder = decoder
        #     pred: List[str] = self.ids_to_clean_text([generated_id])
        #     preds.append(pred[0])
        #     target: List[str] = self.ids_to_clean_text([tgt_id])
        #     targets.append(target[0])
        # self.tokenizer.encoder, self.tokenizer.decoder = ori_encoder, ori_decoder

        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # rouge: Dict = self.calc_generative_metrics(source, preds, target)
        rouge: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, source=source, **rouge)
        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> Seq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)

        if self.hparams.sortish_sampler and type_path != "test" and type_path != "val":
            sampler = dataset.make_sortish_sampler(batch_size, distributed=self.hparams.gpus > 1)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler,
            )

        elif self.hparams.max_tokens_per_batch is not None and type_path != "test" and type_path != "val":
            batch_sampler = dataset.make_dynamic_sampler(
                self.hparams.max_tokens_per_batch, distributed=self.hparams.gpus > 1
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=dataset.collate_fn,
                # shuffle=False,
                num_workers=self.num_workers,
                # batch_size=None,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.num_workers,
                sampler=None,
            )

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        parser.add_argument(
            "--max_source_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=56,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=142,  # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=142,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument("--overwrite_output_dir", action="store_true", default=False)
        parser.add_argument("--max_tokens_per_batch", type=int, default=None)
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=500, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument(
            "--task", type=str, default="summarization", required=False, help="# examples. -1 means use all."
        )
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--src_lang", type=str, default="", required=False)
        parser.add_argument("--tgt_lang", type=str, default="", required=False)
        parser.add_argument("--eval_beams", type=int, default=None, required=False)
        parser.add_argument(
            "--val_metric", type=str, default=None, required=False, choices=["bleu", "rouge2", "loss", None]
        )
        parser.add_argument("--eval_max_gen_length", type=int, default=None, help="never generate more than n tokens")
        parser.add_argument("--save_top_k", type=int, default=1, required=False, help="How many checkpoints to save")
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        return parser


class TranslationModule(SummarizationModule):
    mode = "translation"
    # loss_names = ["loss", "generate_loss", "copy_loss", "copy_or_generate_loss"]
    loss_names = ["loss", "generate_loss", "cls_loss"]
    metric_names = ["bleu"]
    default_val_metric = "bleu"

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.tokenizer = BartTokenizer(vocab_file="tokenizer/vocab.json", merges_file="tokenizer/merges.txt")
        self.dataset_kwargs["src_lang"] = hparams.src_lang
        self.dataset_kwargs["tgt_lang"] = hparams.tgt_lang
        # self.output_dir = hparams.output_dir


    @staticmethod
    def print_first_example_scores(evaluation_dict, num_examples):
        for i in range(num_examples):
            print("evaluating example #{}".format(i))
            print("\tsource (question): {}".format(evaluation_dict["question"][i]))
            print("\tprediction (decomposition): {}".format(evaluation_dict["prediction"][i]))
            print("\ttarget (gold): {}".format(evaluation_dict["gold"][i]))
            print("\texact match: {}".format(round(evaluation_dict["exact_match"][i], 3)))
            print("\tmatch score: {}".format(round(evaluation_dict["match"][i], 3)))
            print("\tstructural match score: {}".format(round(evaluation_dict["structural_match"][i], 3)))
            print("\tsari score: {}".format(round(evaluation_dict["sari"][i], 3)))
            print("\tGED score: {}".format(
                round(evaluation_dict["ged"][i], 3) if evaluation_dict["ged"][i] is not None
                else '-'))
            print("\tstructural GED score: {}".format(
                round(evaluation_dict["structural_ged"][i], 3) if evaluation_dict["structural_ged"][i] is not None
                else '-'))
            print("\tGED+ score: {}".format(
                round(evaluation_dict["ged_plus"][i], 3) if evaluation_dict["ged_plus"][i] is not None
                else '-'))

    @staticmethod
    def print_score_stats(evaluation_dict):
        print("\noverall scores:")

        for key in evaluation_dict:
            # ignore keys that do not store scores
            if key in ["question", "gold", "prediction"]:
                continue
            score_name, scores = key, evaluation_dict[key]

            # ignore examples without a score
            # if None in scores:
            #     scores_ = [score for score in scores if score is not None]
            # else:
            #     scores_ = scores

            # mean_score, max_score, min_score = np.mean(scores_), np.max(scores_), np.min(scores_)
            # print("{} score:\tmean {:.3f}\tmax {:.3f}\tmin {:.3f}".format(
            #     score_name, mean_score, max_score, min_score))
            print("{} score:\tmean {:.3f}".format(
                score_name, scores))

    @staticmethod
    def write_evaluation_output(output_path_base, num_examples, **kwargs):
        # write evaluation summary
        with open(output_path_base + '_summary.tsv', 'w') as fd:
            fd.write('\t'.join([key for key in sorted(kwargs.keys())]) + '\n')
            for i in range(num_examples):
                fd.write('\t'.join([str(kwargs[key][i]) for key in sorted(kwargs.keys())]) + '\n')

        # write evaluation scores per example
        df = pd.DataFrame.from_dict(kwargs, orient="columns")
        df.to_csv(output_path_base + '_full.tsv', sep='\t', index=False)

    def calc_generative_metrics(self, preds, target) -> dict:
        return calculate_bleu(preds, target)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:

        torch.cuda.empty_cache()
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        def special_token(sen):
            replace_dic = {"#10": "ú", "#11": "û", "#12": "ü", "#13": "ý", "#14": "þ", "#15": "ÿ", "#1": "À", "#2": "Á", "#3": "ñ", "#4": "ò", "#5": "õ", "#6": "ö", "#7": "÷", "#8": "ø", "#9": "ù"}
            for k, v in replace_dic.items():
                sen = sen.replace(v, k)
            return sen

        preds = flatten_list([x["preds"] for x in outputs])
        # preds = [special_token(x) for x in preds]
        target = flatten_list([x["target"] for x in outputs])
        # target = [special_token(x) for x in target]
        source = flatten_list([x["source"] for x in outputs])
        if prefix == "test":
            print(len(preds))
            print("*" * 100)
            name = ["question", "preds", "gold"]
            csv_pd = pd.DataFrame(columns=name, data=zip(source, preds, target))
            csv_pd.to_csv("test_pred_res/" + str(self.output_dir).split("/")[-1] + "_cases_study.csv", index=False)

            preds = [x.split("<\s>")[-1].strip() for x in preds]
            target = [x.split("<\s>")[-1].strip() for x in target]

            with open("test_pred_res/" + str(self.output_dir).split("/")[-1] + "_test_preds.csv", 'w') as f, open("test_pred_res/" + str(self.output_dir).split("/")[-1] + "_preds.target", 'w') as semif:
                csv_write = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE,  quotechar='') # , escapechar = ','
                csv_write.writerow(["decomposition"])
                for p in preds:
                    # p = p.replace('"', '').replace(",", "").strip()
                    p = '"' + p.replace('"', '').strip() + '"'
                    csv_write.writerow([p])
                    semif.write(p + "\n")

        scores: Dict = self.calc_break_metrics(source, preds, target)
        print(scores)

        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        # loss, generate_loss, copy_loss, copy_or_generate_loss = losses["loss"], losses["generate_loss"], losses["copy_loss"], losses["copy_or_generate_loss"]
        loss, generate_loss, cls_loss = losses["loss"], losses["generate_loss"], losses["cls_loss"]
        generative_metrics = {
            k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
        }
        metric_val = (
            generative_metrics[self.val_metric] if self.val_metric in generative_metrics else losses[self.val_metric]
        )
        metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
        generative_metrics.update({k: v.item() for k, v in losses.items()})
        losses.update(generative_metrics)
        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["step_count"] = self.step_count
        all_metrics.update(**scores)
        self.metrics[prefix].append(all_metrics)  # callback writes this to self.metrics_save_path

        self.tensorboard_writer.add_scalar("loss", loss, global_step=self.step_count)
        self.tensorboard_writer.add_scalar("cls_loss", cls_loss, global_step=self.step_count)
        self.tensorboard_writer.add_scalar("generate_loss", generate_loss, global_step=self.step_count)
        # self.tensorboard_writer.add_scalar("copy_loss", copy_loss, global_step=self.step_count)
        # self.tensorboard_writer.add_scalar("copy_or_generate_loss", copy_or_generate_loss, global_step=self.step_count)

        for metric, value in all_metrics.items():
            self.tensorboard_writer.add_scalar(metric, value, global_step=self.step_count)

        return {
            "log": all_metrics,
            "preds": preds,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": metric_tensor,
        }

    def calc_break_metrics(self, questions, decompositions, golds) -> dict:

        def replace_nums(s):
            s = re.sub(r'#(\d+)', r'@@\1@@', s)
            # s = s.replace("return ", "")
            return s

        bleu_score = calculate_bleu(decompositions, golds)
        decompositions = [Decomposition(replace_nums(d).split(" ;")) for d in decompositions]
        golds = [Decomposition(replace_nums(g).split(" ;")) for g in golds]
        decompositions_str = [d.to_string() for d in decompositions]
        golds_str = [g.to_string() for g in golds]

        # calculating exact match scores
        exact_match = [d.lower() == g.lower() for d, g in zip(decompositions_str, golds_str)]

        # evaluate using SARI
        sources = [q.split(" ") for q in questions]
        predictions = [d.split(" ") for d in decompositions_str]
        targets = [[g.split(" ")] for g in golds_str]
        sari, keep, add, deletion = get_sari(sources, predictions, targets)

        # evaluate using sequence matcher
        sequence_scorer = SequenceMatchScorer(remove_stop_words=False)
        match_ratio = sequence_scorer.get_match_scores(decompositions_str, golds_str, processing="base")
        structural_match_ratio = sequence_scorer.get_match_scores(decompositions_str, golds_str, processing="structural")

        # evaluate using graph distances
        graph_scorer = GraphMatchScorer()
        decomposition_graphs = [d.to_graph() for d in decompositions]
        gold_graphs = [g.to_graph() for g in golds]

        ged_scores = graph_scorer.get_edit_distance_match_scores(decomposition_graphs, gold_graphs)
        structural_ged_scores = graph_scorer.get_edit_distance_match_scores(decomposition_graphs, gold_graphs, structure_only=True)
        ged_plus_scores = get_ged_plus_scores(decomposition_graphs, gold_graphs, exclude_thr=5, num_processes=10)

        evaluation_dict = {
            # "question": questions,
            # "gold": golds_str,
            # "prediction": decompositions_str,
            "exact_match": np.mean([1 if x else 0 for x in exact_match]),
            "match": np.mean([m for m in match_ratio if m]),
            "structural_match": np.mean([s for s in structural_match_ratio if s]),
            "sari": np.mean([s for s in sari if s]),
            "ged": np.mean([g for g in ged_scores if g]),
            "structural_ged": np.mean([s for s in structural_ged_scores if s]),
            "ged_plus": np.mean([g for g in ged_plus_scores if g])
        }
        # num_examples = len(questions)
        # self.print_first_example_scores(evaluation_dict, min(5, num_examples))
        # self.print_score_stats(evaluation_dict)
        # print("skipped {} examples when computing ged.".format(
        #     len([score for score in ged_scores if score is None])))
        # print("skipped {} examples when computing structural ged.".format(
        #     len([score for score in structural_ged_scores if score is None])))
        # print("skipped {} examples when computing ged plus.".format(
        #     len([score for score in ged_plus_scores if score is None])))

        # if output_path_base:
        #     self.write_evaluation_output(output_path_base, num_examples, **evaluation_dict)

        # if metadata is not None:
        #     metadata = metadata[metadata["question_text"].isin(evaluation_dict["question"])]
        #     metadata["dataset"] = metadata["question_id"].apply(lambda x: x.split("_")[0])
        #     metadata["num_steps"] = metadata["decomposition"].apply(lambda x: len(x.split(";")))
        #     score_keys = [key for key in evaluation_dict if key not in ["question", "gold", "prediction"]]
        #     for key in score_keys:
        #         metadata[key] = evaluation_dict[key]
        #
        #     for agg_field in ["dataset", "num_steps"]:
        #         df = metadata[[agg_field] + score_keys].groupby(agg_field).agg("mean")
        #         print(df.round(decimals=3))

        evaluation_dict.update(bleu_score)
        # print(evaluation_dict)
        return evaluation_dict


def main(args, model=None) -> SummarizationModule:
    Path(args.output_dir).mkdir(exist_ok=True)
    writer = SummaryWriter(args.output_dir + "/run")
    check_output_dir(args, expected_items=3)

    if model is None:
        if "summarization" in args.task:
            model: SummarizationModule = SummarizationModule(args)
        else:
            model: SummarizationModule = TranslationModule(args)
    model.tensorboard_writer = writer
    dataset = Path(args.data_dir).name
    if (
        args.logger_name == "default"
        or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        logger = True  # don't pollute wandb logs unnecessarily
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        project = os.environ.get("WANDB_PROJECT", dataset)
        logger = WandbLogger(name=model.output_dir.name, project=project)

    elif args.logger_name == "wandb_shared":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(name=model.output_dir.name, project=f"hf_{dataset}")

    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    else:
        es_callback = False

    lower_is_better = args.val_metric == "loss"
    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(
            args.output_dir, model.val_metric, args.save_top_k, lower_is_better
        ),
        early_stopping_callback=es_callback,
        logger=logger,
    )
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")
    if not args.do_predict:
        return model

    model.hparams.test_checkpoint = ""
    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[0]
        trainer.resume_from_checkpoint = checkpoints[0]
    trainer.logger.log_hyperparams(model.hparams)

    # test() without a model tests using the best checkpoint automatically
    trainer.test(model=model, ckpt_path=checkpoints[0], verbose=True)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())

    args = parser.parse_args()

    main(args)
