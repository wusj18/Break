import os
import re
import csv
import json
import spacy
import random


class data_format_trans(object):
    def __init__(self, entity_source=None):
        self.nlp = spacy.load('en')
        self.entitys_dic = {}
        if entity_source:
            self.get_entitys_dic(entity_source)

    def special_token(self, sen):
        replace_dic = {"#10": "ú", "#11": "û", "#12": "ü", "#13": "ý", "#14": "þ", "#15": "ÿ", "#16": "Ā", "#17": "ā",
                       "#18": "Ă", "#19": "ă", "#1": "À", "#2": "Á", "#3": "ñ", "#4": "ò", "#5": "õ", "#6": "ö",
                       "#7": "÷", "#8": "ø", "#9": "ù"}
        for k, v in replace_dic.items():
            sen = sen.replace(v, k)
        return sen

    def replace_nums(self, s):
        s = re.sub(r'#(\d)', r'@@\1@@', s)
        # s = s.replace("return ", "")
        return s

    def csv2srctgt(self, data_dir="", csvfile="", enhance_num=0):
        prefix = csvfile.split(".")[0]
        lexicon_file = prefix + "_lexicon_tokens.json"

        with open(data_dir + csvfile, 'r') as f, open(data_dir + prefix + ".source", 'w') as fsource, open(
                data_dir + prefix + ".target", "w") as ftarget, open(data_dir + lexicon_file, "r",
                                                                     encoding="utf-8") as flex, open(
                data_dir + lexicon_file.rstrip(".json") + "_special.json", "w", encoding="utf-8") as foutlex:
            freq_words = self.get_lexicon_freq_words(data_dir + lexicon_file, prefix=prefix)
            word_file = flex.readlines()
            reader = csv.reader(f)
            sub_ques_num_dic = {}
            lines_num = 0
            enhanced_lines_num = 0
            induction_graph = {}
            for row in reader:
                if lines_num == 0:
                    lines_num += 1
                    continue
                word_line = word_file[lines_num - 1]
                words_line = json.loads(word_line)
                if len(row) == 1:
                    src = row[0]
                    tgt = row[0]
                else:
                    # src = self.special_token(row[1])
                    src = row[1]
                    tgt = re.sub(r'@@(\d?)@@', r'#\1', row[2])
                    # tgt = self.special_token(tgt)  # .split(";")
                ori_words = sorted(words_line["allowed_tokens"][1:-1].replace(" '", "").replace("'", "").split(","))
                ori_words = [x for x in ori_words if x not in freq_words]
                if enhance_num:
                    enhanced_src, enhanced_tgt, enhanced_words = self.data_enhance(src, tgt, ori_words, enhance_num)
                    enhanced_src += [src]
                    enhanced_tgt += [tgt]
                    enhanced_words += [ori_words]
                else:
                    enhanced_src, enhanced_tgt, enhanced_words = [src], [tgt], [ori_words]
                assert len(enhanced_src) == len(enhanced_tgt)

                for src, tgt, sword in zip(enhanced_src, enhanced_tgt, enhanced_words):
                    new_words_line = words_line
                    new_words_line["allowed_tokens"] = sword
                    # src = re.sub(r'#\d+', "", src)
                    if not enhance_num and new_words_line["source"] != src:
                        print("WARNING: 行未对齐！")
                        print(src, new_words_line["source"])
                        src = new_words_line["source"]
                    if enhance_num:
                        new_words_line["source"] = src
                    foutlex.write(json.dumps(new_words_line) + "\n")
                    if len(row) > 3:
                        opt = row[3][1:-1].replace("'", "").split(", ")

                        symb_re = re.compile(r'#\d+')
                        sub_questions = tgt.split(";")
                        for index, sub_q in enumerate(sub_questions):
                            sym = re.findall(symb_re, sub_q)
                            if sym:
                                opt[index] += " " + " ".join(sym)

                        # ftarget.write(" ".join(opt) + " <\s> " + tgt + "\n")
                        opt = " ".join(opt)
                        if opt in induction_graph:
                            induction_graph[opt] += 1
                        else:
                            induction_graph[opt] = 1

                        fsource.write(src + " <\s> " + opt + "\n")
                        ftarget.write(tgt + "\n")
                    else:
                        fsource.write(src + "\n")
                        ftarget.write(tgt + "\n")
                    if enhanced_lines_num % 1000 == 0:
                        print(str(enhanced_lines_num) + "cases has been generated!")
                    enhanced_lines_num += 1
                lines_num += 1

                sub_ques = len(tgt.split(";"))
                if sub_ques in sub_ques_num_dic.keys():
                    sub_ques_num_dic[sub_ques] += 1
                else:
                    sub_ques_num_dic[sub_ques] = 1

            print(prefix + " " + str(enhanced_lines_num))
            # print(sorted(sub_ques_num_dic.items(), key=lambda d: d[1]))
            # induction_graph = [x for x in induction_graph.items() if x[1] > 50]
            # # if x[1] > 20
            # sum_num = sum([x[1] for x in induction_graph])
            # print(len(induction_graph), sum_num)
            # print(sorted(induction_graph, key=lambda d: d[1]))

    def case_study(self, data_dir, csvfile):
        prefix = csvfile.split(".")[0]
        miss_words_dic = {}
        with open(data_dir + csvfile, 'r') as f:
            reader = csv.reader(f)
            # print(type(reader))
            lines_num = 0
            for row in reader:
                if lines_num == 0:
                    lines_num += 1
                    continue
                pred_words = self.special_token(row[1]).split(" ")
                target_words = self.special_token(row[2]).split(" ")
                # miss_words = [x for x in target_words if x not in pred_words]
                miss_words = [x for x in pred_words if x not in target_words]

                miss_words = list(set(miss_words))
                for w in miss_words:
                    if w not in miss_words_dic:
                        miss_words_dic[w] = 1
                    else:
                        miss_words_dic[w] += 1
            print(sorted(miss_words_dic.items(), key=lambda d: d[1]))

    def get_lexicon_freq_words(self, file_name, prefix="train", tgt_file=None):
        seen = {}
        with open(file_name, "r", encoding="utf-8") as f:
            # , open(tgt_file) as tgtf:
            # lines = json.load(f)
            # for line, tgt in zip(f, tgtf):
            for line in f:
                line = json.loads(line)
                words = line["allowed_tokens"][1:-1].replace(" '", "").replace("'", "").split(",")
                # words = line["allowed_tokens"]
                # ques = line["source"]
                # tgt = tgt.strip().split(" ")
                for w in words:
                    if w in seen:
                        seen[w] += 1
                    else:
                        seen[w] = 1
            words_stat = sorted(seen.items(), key=lambda x: x[1], reverse=True)
            if prefix == "train":
                words_stat = [x[0] for x in words_stat if x[1] >= 3100]
            else:
                words_stat = [x[0] for x in words_stat if x[1] >= 600]
            words_freq = sorted(words_stat)
        return words_freq

    def spacy_csv(self, data_dir, csvfile):
        prefix = csvfile.split(".")[0]
        with open(data_dir + csvfile, 'r') as f:
            reader = csv.reader(f)
            lines_num = 0
            for row in reader:
                if lines_num == 0:
                    lines_num += 1
                    continue
                if len(row) < 1:
                    continue
                question = row[1]
                decomp = row[2]

                question = self.nlp(question)
                decomp = self.nlp(decomp)
                for token in question:
                    print(token, token.pos_, token.pos)
                print("-" * 20)
                for token in decomp:
                    print(token, token.pos_, token.pos)
                print("*" * 20)
                for token in question.ents:
                    print(token, token.label_, token.label)
                print("-" * 20)
                for token in decomp.ents:
                    print(token, token.label_, token.label)
                print("*" * 20)
                for token in question.noun_chunks:
                    print(token)
                print("-" * 20)
                for token in decomp.noun_chunks:
                    print(token)
                print("*" * 20)

                break

    def get_entitys_dic(self, file):
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                question = self.nlp(line)
                for token in question.ents:
                    if token.label_ not in self.entitys_dic:
                        self.entitys_dic[token.label_] = [str(token).strip()]
                    else:
                        self.entitys_dic[token.label_] += [str(token).strip()]
        print("Entity dic has been built!")

    def data_enhance(self, question, tgt, words, enhance_num):
        question = re.sub(' +', ' ', str(question))
        tgt = re.sub(' +', ' ', str(tgt))
        question_ent = self.nlp(question)
        tgt_ent = self.nlp(tgt)

        if len(question_ent.ents) < 1:
            return ([question], [tgt], [words])

        replace_ents = []
        replace_ents_labels = []
        tgt_ents_str = [re.sub(' +', ' ', str(x)) for x in tgt_ent.ents]
        for ent in question_ent.ents:
            cur_ent = re.sub(' +', ' ', str(ent))
            if cur_ent in tgt_ents_str:
                replace_ents += [ent]
                replace_ents_labels += [ent.label_]
        if len(replace_ents) < 1:
            return ([question], [tgt], [words])

        enhanced_questions = []
        enhanced_tgt = []
        enhanced_words = []
        for _ in range(enhance_num):
            question_new = question
            tgt_new = tgt
            words_new = list(words[:])
            for ent, ent_label in zip(replace_ents, replace_ents_labels):
                # seed = random.randint(0, 1)
                # if seed > 0.8:
                #     continue
                # else:
                length = len(self.entitys_dic[ent_label])
                seed = random.randint(1, length - 1)
                new_ent = str(self.entitys_dic[ent_label][seed])
                question_new = question_new.replace(str(ent), new_ent)
                tgt_new = tgt_new.replace(str(ent), new_ent)
                for w in str(ent).split(" "):
                    if w in words_new:
                        index = words_new.index(w)
                        words_new.pop(index)
                words_new = words_new + new_ent.split(" ")
            enhanced_questions += [question_new.replace("\n", "")]
            enhanced_tgt += [tgt_new.replace("\n", "")]
            enhanced_words += [sorted(list(set(words_new)))]
        return (enhanced_questions, enhanced_tgt, enhanced_words)


def data_trans(data_dir, entity_source=None, enhance_num=0):
    dft = data_format_trans(entity_source=entity_source)
    for file in os.listdir(data_dir):
        if file.endswith("csv"):
            dft.csv2srctgt(data_dir, file, enhance_num=enhance_num)
            # if "train.csv" == file:
            # dft.case_study(data_dir, file)
            #     dft.spacy_csv(data_dir, file)
        # if file.endswith("json") and "sorted" not in file:
        #     dft.lexicon_words(data_dir + file, data_dir + file.split("_")[0] + ".target")


def case_study_induction_graph(data_dir, csvfile):
    with open(data_dir + csvfile, 'r') as f:
        reader = csv.reader(f)
        lines_num = 0
        true_case = 0
        for row in reader:
            if lines_num == 0:
                lines_num += 1
                continue
            if row[1].split("<\s>")[0] == row[2].split("<\s>")[0]:
                true_case += 1
            lines_num += 1
    print(true_case, lines_num, true_case / lines_num)


if __name__ == '__main__':
    # case_study_induction_graph("../../../test_pred_res/", "3y19_8e5_base_optssym_enhance_cases_study.csv")
    data_trans("../QDMR_golden_graph/")
    # , entity_source="../../umt_training_data/val.source", enhance_num = 2
