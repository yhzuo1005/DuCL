import argparse
import time
import logging
import os
import sys
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def logging_config_fn(logging_dir="./logs",
                      logging_filename="logfile",
                      logging_level=logging.DEBUG,
                      print_on_screen=True):

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    logging_path = os.path.join(logging_dir, logging_filename+"_"+str(datetime.now())[:10]+".txt")
    output_format = "[%(asctime)s] {%(levelname)s} (%(message)s)"
    if print_on_screen:
        logging.basicConfig(level=logging_level,
                            format=output_format,
                            datefmt="%Y-%m-%d %H:%M:%S",
                            handlers=[logging.FileHandler(logging_path),
                                      logging.StreamHandler(sys.stdout)]
                            )
    else:
        logging.basicConfig(filename=logging_path,
                            level=logging_level,
                            format=output_format,
                            datefmt="%Y-%m-%d %H:%M:%S",
                            )


# 种子
def init_seed(seed=None):
    if seed is None:
        seed = int(time.time()*1000//1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

# 指标计算
def Metrics(trues, predicts):
    precision = precision_score(trues, predicts)
    recall = recall_score(trues, predicts)
    f1 = f1_score(trues, predicts)
    acc = accuracy_score(trues, predicts)
    return precision, recall, f1, acc





def get_parser():
    # 指定data为BQ
    data = "BQ"
    parser = argparse.ArgumentParser(f"dataset is {data}---contrastive learning & block interaction...")
    parser.add_argument("--data", default=f"{data}", type=str)

    parser.add_argument("--device", default=0, type=int, help="0 for gpu, -1 for cpu, default is gpu")
    parser.add_argument("--seed", default=42, type=int)

    # 预训练模型
    parser.add_argument("--pretrained_model_path", default="./bert_chinese", help="bert_base模型路径")

    parser.add_argument("--train_data", default=f"./dataset/{data}/{data}.train.data", type=str, help=f"{data} train data")
    parser.add_argument("--dev_data", default=f"./dataset/{data}/{data}.valid.data", type=str, help=f"{data} dev data")
    parser.add_argument("--test_data", default=f"./dataset/{data}/{data}.test.data", type=str, help=f"{data} test data")


    # 三类模型
    # 句子级别的对比学习
    parser.add_argument("--sent_model_dir", default="./sent_model", type=str, help="the dir of sent_model")
    # 句子对级别的对比学习
    parser.add_argument("--pair_model_dir", default="./pair_model", type=str, help="the dir of pair_model")
    # 分类器
    parser.add_argument("--classifier_dir", default="./classifier", type=str, help="the dir of classifier")

    # 日志文件
    # 日志文件夹名 # 日志文件名
    parser.add_argument("--sent_log_dir", default="./sent_logs", type=str, help="the dir of sent_log")
    parser.add_argument("--sent_log_file", default="./sent_logs", type=str, help="the filename")

    parser.add_argument("--pair_log_dir", default="./pair_logs", type=str, help="the dir of pair_log")
    parser.add_argument("--pair_log_file", default="./pair_logs", type=str, help="the filename")

    parser.add_argument("--classifier_log_dir", default="./classifier_logs", type=str, help="the dir of classifier_log")
    parser.add_argument("--classifier_log_file", default="./classifier_logs", type=str, help="the filename")


    # sent+pair+classifier的日志文件
    parser.add_argument("--train_log_dir", default="./train_logs", type=str, help="all the logs in train stage.")
    parser.add_argument("--train_log_file", default="./train_logs", type=str, help="the filename")


    parser.add_argument("--test_log_dir", default="./test_logs", type=str, help="all the logs in test stage.")
    parser.add_argument("--test_log_file", default="./test_logs", type=str, help="the filename")

    # 参数
    parser.add_argument("--num_epoch_pair", default=50, type=int, help=f"the number of epochs of pair_model on BQ is 4.")




    # # stage1的温度系数
    # parser.add_argument("--temp_unsup", default=0.05, type=float, help="the temperature of simcse_unsup_loss.")



    # stage2的温度系数
    parser.add_argument("--temp_sup", default=0.05, type=float, help="the temperature of SupConLoss.")


    parser.add_argument("--load_sent_model", default=True, type=bool, help="if true, load sent_model in stage2; if false, don't load sent_model in stage2.")


    parser.add_argument("--num_sent", default=10000, type=int, help="the number of sentences in sent_model")
    parser.add_argument("--num_epoch_classifier", default=50, type=int, help="the number of epochs of classifier")
    parser.add_argument("--num_patience", default=3, type=int, help="the number of patience of classifier")

    parser.add_argument("--label_size", default=2, help="2-classes")
    parser.add_argument("--lr_model", default=5e-5, type=float, help="lr_model")

    parser.add_argument("--max_len", default=30, type=int)

    parser.add_argument("--num_features", default=128, type=int, help="the number of features in pair loss")

    # 暂时不管temp
    # parser.add_argument("--temp", default=0.05, type=float)


    parser.add_argument("--batch_size", default=32,type=int, required=True)
    # stage1的温度系数
    parser.add_argument("--temp_unsup", default=0.05, type=float, help="the temperature of simcse_unsup_loss.", required=True)
    parser.add_argument("--lr_bert", default=2e-5, type=float, help=f"lr_bert on {data} is 2e-5", required=True)
    parser.add_argument("--dropout_rate", default=0.3, type=float, required=True)

    parser.add_argument("--balance", default=0.1, type=float, required=True)


    args = parser.parse_args()

    return args

# if __name__ == '__main__':
#     args = get_parser()
#     print(args)


