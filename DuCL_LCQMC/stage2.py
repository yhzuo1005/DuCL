#coding:utf-8
import logging
import os
import time

import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from config import get_parser, init_seed, logging_config_fn
from data_process import supDataset, load_sup_data
from model import CLmodel, BasicBert
from utils import SupConLoss, train_sup, dev_sup


def train_pair_model(args, params):
    start = time.time()

    # args = get_parser()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    init_seed(args.seed)

    # 日志文件
    logging_config_fn(logging_dir=args.pair_log_dir,
                      logging_filename=args.pair_log_file,
                      logging_level=logging.INFO,
                      print_on_screen=False)

    # # 记录模型参数
    # logging.info("======logs of parameters======")
    # message = '\n'.join([f'{k:<30}: {v}' for k, v in vars(args).items()])
    # logging.info(message)

    # 模型存放路径，如果不存在则创建。
    if not os.path.exists(args.pair_model_dir):
        os.makedirs(args.pair_model_dir)

    logging.info("========pair_model begin=========")
    logging.info(f"Dataset is {args.data}")

    # 分词器
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)

    # 训练集
    sup_train = load_sup_data(args.train_data)
    sup_dataset_train = supDataset(data=sup_train, tokenizer=tokenizer, max_len=args.max_len)
    sup_dataloader_train = DataLoader(dataset=sup_dataset_train, batch_size=params["batch_size"], shuffle=True)

    # 验证集
    sup_dev = load_sup_data(args.dev_data)
    sup_dataset_dev = supDataset(data=sup_dev, tokenizer=tokenizer, max_len=args.max_len)
    sup_dataloader_dev = DataLoader(dataset=sup_dataset_dev, batch_size=params["batch_size"], shuffle=True)


    # pair_model 初始化
    model_sup = CLmodel(is_sentence_level=False,
                          encoder=BasicBert.from_pretrained(
                          args.pretrained_model_path,
                          num_labels=params["num_features"],
                          attention_probs_dropout_prob=params["dropout_rate"],
                          hidden_dropout_prob=params["dropout_rate"],
                          output_attentions=False,
                          output_hidden_states=False
                          ),
                          is_train=True,
                          pooling=None
                          ).to(device=args.device)

    # 是否加载sent_model
    if args.load_sent_model:
        logging.info(f"loading the sent_model in stage2...")
        checkpoint = torch.load(os.path.join(args.sent_model_dir, "best.pth.tar"))
        model_sup.load_state_dict(checkpoint["model"])

    # 损失函数 ce_loss, pair_loss
    criterion_ce = nn.CrossEntropyLoss()
    criterion_sup = SupConLoss(params["temp_sup"])

    # 优化器
    optimizer = torch.optim.AdamW(model_sup.parameters(), params["lr_bert"])

    start_epoch, end_epoch = 1, args.num_epoch_pair
    best_acc = 0.0
    patience_counter = 0


    for epoch in range(start_epoch, end_epoch+1):
        logging.info(f"train pair_model in epoch {epoch}:")
        epoch_time, epoch_loss = train_sup(model_sup, sup_dataloader_train, optimizer, criterion_ce, criterion_sup, args.device, params["balance"])
        logging.info(f"each epoch train time is {(epoch_time / 60):.2f}, loss is {epoch_loss:.4f}.")

        logging.info(f"dev pair_model in epoch: {epoch}:")
        dev_time, dev_loss, dev_precision, dev_recall, dev_f1, dev_acc = dev_sup(model_sup, sup_dataloader_dev, criterion_ce, criterion_sup, args.device, params["balance"])
        logging.info(f"each epoch dev time is {(dev_time/60):.2f}, dev loss is {dev_loss:.4f}.")
        logging.info(f"dev precision is {dev_precision}, dev recall is {dev_recall}, dev f1 is {dev_f1}, dev acc is {dev_acc}.")

        if dev_acc < best_acc:
            patience_counter += 1
        else:
            # 保存在验证集上性能最好的模型
            logging.info(f"save the BEST model.")
            best_acc = dev_acc
            patience_counter = 0
            torch.save({"epoch": epoch,
                        "loss": dev_loss,
                        "model": model_sup.state_dict(),
                        "best_acc": best_acc
                        }, os.path.join(args.pair_model_dir, "best.pth.tar"))

        # 保存每一个epoch的模型
        logging.info(f"save model in epoch {epoch}.")
        torch.save({"epoch": epoch,
                    "loss": dev_loss,
                    "model": model_sup.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_acc": best_acc
                    }, os.path.join(args.pair_model_dir, f"model_{epoch}.pth.tar"))


        if patience_counter >= args.num_patience:
            logging.info(f"early stopping: patience number {args.num_patience} limit reached, so early stopping!")
            break

    # # 自动调参，报道的结果一定是数值而不是张量等其他类型。
    # nni.report_final_result(best_acc)


    time_cost = time.time() - start
    logging.info(f"TRAIN&DEV pair_model costs time {time_cost / 60:.2f} minutes.")
    logging.info("========pair_model end=========")


if __name__ == '__main__':
    args = get_parser()

    # 1.设定要调参数的默认值。
    params_pair = {"lr_bert": args.lr_bert,
              "balance": args.balance,
              "batch_size": args.batch_size,
              "dropout_rate": args.dropout_rate,
              "temp_sup": args.temp_sup,
              "num_features": args.num_features
              }

    # # 2.获取搜索空间中的超参数
    # optimized_params = nni.get_next_parameter()
    # params_pair.update(optimized_params)

    # 训练
    train_pair_model(args, params_pair)

