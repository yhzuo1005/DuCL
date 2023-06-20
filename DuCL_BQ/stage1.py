import logging
import os
import random
import time
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW
from config import get_parser, logging_config_fn, init_seed
from data_process import load_unsup_data, load_sup_data, unsupDataset, supDataset,load_extract_data
from model import CLmodel, BasicBert
from utils import simcse_unsup_loss, train_unsup, test_unsup


def train_sent_model(args):
    start = time.time()

    # args = get_parser()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # 日志文件
    logging_config_fn(logging_dir=args.sent_log_dir,
                      logging_filename=args.sent_log_file,
                      logging_level=logging.INFO,
                      print_on_screen=False)
    # 记录模型参数
    logging.info("======logs of parameters======")
    message = '\n'.join([f'{k:<30}: {v}' for k, v in vars(args).items()])
    logging.info(message)

    init_seed(args.seed)

    # 模型存放路径，如果不存在就创建。
    if not os.path.exists(args.sent_model_dir):
        os.makedirs(args.sent_model_dir)


    logging.info("========sent_model begin=========")
    logging.info(f"Dataset is {args.data}")

    # 分词器
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)


    # # unsup_data
    # unsup_sent1, unsup_sent2, unsup_sent, unsup_label = load_unsup_data(args.train_data, args.dev_data, args.test_data)
    # # 打乱数据
    # unsup_sent_shuffle = random.sample(unsup_sent, args.num_sent)

    # unique_sent
    unsup_sent_shuffle = load_extract_data("./dataset/BQ/BQ.sent_unique.txt")


    # sup_data
    sup_train = load_sup_data(args.train_data)
    sup_dev = load_sup_data(args.dev_data)
    sup_test = load_sup_data(args.test_data)


    # 无监督数据集的dataloader
    unsup_dataset = unsupDataset(data=unsup_sent_shuffle, tokenizer=tokenizer, max_len=args.max_len)
    unsup_dataloader = DataLoader(dataset=unsup_dataset, batch_size=args.batch_size, shuffle=True)


    # 有监督训练集
    sup_dataset_train = supDataset(data=sup_train, tokenizer=tokenizer, max_len=args.max_len)
    sup_dataloader_train = DataLoader(dataset=sup_dataset_train, batch_size=args.batch_size, shuffle=True)


    # 有监督验证集
    sup_dataset_dev = supDataset(data=sup_dev, tokenizer=tokenizer, max_len=args.max_len)
    sup_dataloader_dev = DataLoader(dataset=sup_dataset_dev, batch_size=args.batch_size, shuffle=True)


    # 有监督测试集
    sup_dataset_test = supDataset(data=sup_test, tokenizer=tokenizer, max_len=args.max_len)
    sup_dataloader_test = DataLoader(dataset=sup_dataset_test, batch_size=args.batch_size, shuffle=False)


    # 句子级别的无监督模型
    model_unsup = CLmodel(is_sentence_level=True,
                          encoder=BasicBert.from_pretrained(
                              args.pretrained_model_path,
                              num_labels=args.num_features,
                              attention_probs_dropout_prob=args.dropout_rate,
                              hidden_dropout_prob=args.dropout_rate,
                              output_attentions=False,
                              output_hidden_states=False
                          ),
                          is_train=False,
                          pooling=None
    ).to(device=args.device)


    # 优化器
    optimizer = torch.optim.AdamW(model_unsup.parameters(), args.lr_bert)
    # optimizer = AdamW(model_unsup.parameters(), params["lr_bert"])

    corrcoef_devdata = train_unsup(model_unsup, unsup_dataloader, sup_dataloader_dev, simcse_unsup_loss, args.temp_unsup, optimizer, args.device, args.sent_model_dir)
    logging.info(f"-> best_corrcoef in dev_data is {corrcoef_devdata:.4f}.")


    logging.info(f"test model_unsup at {args.data}-dataset-train, {args.data}-dataset-dev, {args.data}-dataset-test.")
    corrcoef_train, corrcoef_dev, corrcoef_test = test_unsup(args.pretrained_model_path, args.num_features, args.dropout_rate, args.sent_model_dir, args.device, sup_dataloader_train, sup_dataloader_dev, sup_dataloader_test)
    logging.info(f"corrcoef in train dataset is {corrcoef_train:.4f}, corrcoef in dev dataset is {corrcoef_dev:.4f}, corrcoef in test dataset is {corrcoef_test:.4f}.")


    time_cost = time.time()-start
    logging.info(f"train sent_model costs time {time_cost/60:.4f} minutes.")
    logging.info("========sent_model end=========")



if __name__ == '__main__':
    args = get_parser()

    # 训练
    train_sent_model(args)


