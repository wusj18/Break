# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune.sh --help to see all the possible options
export CUDA_VISIBLE_DEVICES=0
python finetune.py --data_dir=dataset/Break-dataset/QDMR_high_opts/  --model_name_or_path=facebook/bart-base  --learning_rate=8e-5  --train_batch_size=24  --eval_batch_size=24  --output_dir=outputs/test  --task=translation  --num_train_epochs=5  --gpus=1  --n_train=10  --n_val=10  --n_test=10  --save_top_k=1  --do_train  --do_predict  --overwrite_output_dir  --gradient_clip_val=5
