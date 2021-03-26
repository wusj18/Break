# Break
Break data and code

# 代码

训练入口：seq2seq/finetune.py

数据读取：seq2seq/utils.py  class AbstractSeq2SeqDataset class Seq2SeqDataset

模型位置：transformers/models/bart/*

inference生成：transformers/generation_utils.py

数据预处理：seq2seq/dataset/data_format_trans.py



# 训练
seq2seq/finetune.sh里面 训练就--do_train 测试就--do_predict



# 数据 （粗粒度break数据 = break high level）

粗粒度Break基本数据，用于本地评估  dataset/break_high_data_val

粗粒度Break基本数据，训练集和验证集整合在一起的，用于提交 dataset/ break_high_data_test_train+val

细粒度Break数据，用于post-train  dataset/Break-dataset/QDMR

外源数据，用于post-train  dataset/umt_training_data和dataset/strategyqa_dataset （但是数据分布不甚一致，目前不好用）



# 版本

pytorch   1.0.0

pytorch-lightning             1.0.6
