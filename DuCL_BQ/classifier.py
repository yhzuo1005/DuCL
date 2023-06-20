import logging
import os
import time

import nni
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from config import get_parser, init_seed, logging_config_fn, Metrics
from data_process import load_sup_data, supDataset
from model import CLmodel, Classifier, BasicBert
from test import test_sup


def train_fn(model_sup, model_classifier, sup_dataloader_train, criterion_ce, optimizer, device):
    model_sup.eval()
    model_classifier.train()

    start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(sup_dataloader_train)):
        batch_start = time.time()
        source, target, label = batch

        source_input_ids = source.get("input_ids").squeeze(1).to(device)
        source_attention_mask = source.get("attention_mask").squeeze(1).to(device)
        source_token_type_ids = source.get("token_type_ids").squeeze(1).to(device)

        target_input_ids = target.get("input_ids").squeeze(1).to(device)
        target_attention_mask = target.get("attention_mask").squeeze(1).to(device)
        target_token_type_ids_pre = target.get("token_type_ids").squeeze(1).to(device)
        target_token_type_ids = torch.ones_like(target_token_type_ids_pre)

        input_ids = (source_input_ids, target_input_ids)
        attention_mask = (source_attention_mask, target_attention_mask)
        token_type_ids = (source_token_type_ids, target_token_type_ids)

        with torch.no_grad():
            feat = model_sup(input_ids, attention_mask, token_type_ids)
        logits, probs = model_classifier(feat.detach())

        optimizer.zero_grad()
        loss = criterion_ce(logits.view(-1, 2), label.to(device).view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        batch_time_avg += time.time()-batch_start
        description = f"Avg batch time is {(batch_time_avg/(batch_idx+1)):.4f}s, loss is {(running_loss/(batch_idx+1)):.4f}."
        tqdm(sup_dataloader_train).set_description(description)

    train_time = time.time()-start
    train_loss = running_loss/len(sup_dataloader_train)
    return train_time, train_loss

def dev_fn(model_sup, model_classifier, sup_dataloader_dev, criterion_ce, device):
    model_sup.eval()
    model_classifier.eval()

    start = time.time()
    running_loss = 0.0
    batch_time_avg = 0.0

    trues, preds = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(sup_dataloader_dev)):
            batch_start = time.time()

            source, target, label = batch
            source_input_ids = source.get("input_ids").squeeze(1).to(device)
            source_attention_mask = source.get("attention_mask").squeeze(1).to(device)
            source_token_type_ids = source.get("token_type_ids").squeeze(1).to(device)

            target_input_ids = target.get("input_ids").squeeze(1).to(device)
            target_attention_mask = target.get("attention_mask").squeeze(1).to(device)
            target_token_type_ids_pre = target.get("token_type_ids").squeeze(1).to(device)
            target_token_type_ids = torch.ones_like(target_token_type_ids_pre)

            input_ids = (source_input_ids, target_input_ids)
            attention_mask = (source_attention_mask, target_attention_mask)
            token_type_ids = (source_token_type_ids, target_token_type_ids)

            feat = model_sup(input_ids, attention_mask, token_type_ids)
            logits, probs = model_classifier(feat.detach())
            loss = criterion_ce(logits.view(-1, 2), label.to(device).view(-1))
            running_loss += loss.item()

            trues.extend(label.numpy())
            preds.extend(probs[:, 1].detach().cpu().numpy())

            batch_time_avg += time.time()-batch_start
            description = f"Avg batch time is {(batch_time_avg/(batch_idx+1)):.4f}s, loss is {(running_loss/(batch_idx+1)):.4f}."
            tqdm(sup_dataloader_dev).set_description(description)
    trues = np.array(trues)
    preds = np.array(preds)

    predict = [1 if x >= 0.5 else 0 for x in preds]
    precision, recall, f1, acc = Metrics(trues, predict)

    dev_time = time.time()-start
    dev_loss = running_loss/len(sup_dataloader_dev)

    return dev_time, dev_loss, precision, recall, f1, acc




def train_classifier(args, params):
    start = time.time()

    # args = get_parser()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    init_seed(args.seed)

    # # 日志文件
    # logging_config_fn(logging_dir=args.classifier_log_dir,
    #                   logging_filename=args.classifier_log_file,
    #                   logging_level=logging.INFO,
    #                   print_on_screen=False)
    #
    # logging.info("======logs of parameters======")
    # message = '\n'.join([f'{k:<30}: {v}' for k, v in vars(args).items()])
    # logging.info(message)

    # 模型存放路径
    if not os.path.exists(args.classifier_dir):
        os.makedirs(args.classifier_dir)


    logging.info("========classifier begin=========")
    logging.info(f"Dataset is {args.data}")

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)

    sup_train = load_sup_data(args.train_data)
    sup_dev = load_sup_data(args.dev_data)
    sup_test = load_sup_data(args.test_data)

    # 训练集
    sup_dataset_train = supDataset(data=sup_train, tokenizer=tokenizer, max_len=args.max_len)
    sup_dataloader_train = DataLoader(dataset=sup_dataset_train, batch_size=params["batch_size"], shuffle=True)

    # 验证集
    sup_dataset_dev = supDataset(data=sup_dev, tokenizer=tokenizer, max_len=args.max_len)
    sup_dataloader_dev = DataLoader(dataset=sup_dataset_dev, batch_size=params["batch_size"], shuffle=True)

    # 测试集
    test_dataset = supDataset(data=sup_test, tokenizer=tokenizer, max_len=args.max_len)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=params["batch_size"], shuffle=False)


    # 初始化模型model_sup
    model_sup = CLmodel(is_sentence_level=False,
                        encoder=BasicBert.from_pretrained(
                        args.pretrained_model_path,
                        num_labels=params["num_features"],
                        attention_probs_dropout_prob=params["dropout_rate"],
                        hidden_dropout_prob=params["dropout_rate"],
                        output_attentions=False,
                        output_hidden_states=False
                        ),
                        is_train=False,
                        pooling=None
                        ).to(device=args.device)

    # 加载最佳模型model_sup
    model_sup_ckpt = torch.load(os.path.join(args.pair_model_dir, "best.pth.tar"))
    model_sup.load_state_dict(model_sup_ckpt["model"])

    # 分类器初始化
    model_classifier = Classifier(
        encoder=BasicBert.from_pretrained(
            args.pretrained_model_path,  # Use the 12-layer BERT model
            num_labels=args.label_size,
            output_attentions=False,
            output_hidden_states=False,
        ),
        num_classes=args.label_size,
        dropout_rate=params["dropout_rate"]
    ).to(args.device)

    criterion_ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model_classifier.parameters(), lr=params["lr_model"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=0)

    start_epoch, end_epoch = 1, args.num_epoch_classifier
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(start_epoch, end_epoch+1):
        # 训练集
        logging.info(f"train classifier in epoch {epoch}")
        train_time, train_loss = train_fn(model_sup, model_classifier, sup_dataloader_train, criterion_ce, optimizer, args.device)
        logging.info(f"train time is {(train_time/60):.2f} minutes, train_loss is {train_loss:.4f}.")

        # 验证集
        logging.info(f"dev classifier in epoch {epoch}")
        dev_time, dev_loss, dev_precision, dev_recall, dev_f1, dev_acc = dev_fn(model_sup, model_classifier, sup_dataloader_dev, criterion_ce, args.device)
        logging.info(f"dev time is {(dev_time/60):.2f} minutes, dev_loss is {dev_loss:.4f}.")
        logging.info(f"====DEV metrics: precision is {dev_precision:.4f}, recall is {dev_recall:.4f}, f1 is {dev_f1:.4f}, acc is {dev_acc:.4f}.===")
        scheduler.step(dev_acc)


        if dev_acc < best_acc:
            patience_counter += 1
        else:
            # 保存最好的模型
            best_acc = dev_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model": model_classifier.state_dict(),
                "loss": dev_loss,
                "best_acc": best_acc,
            }, os.path.join(args.classifier_dir, "best.pth.tar"))

        # 保存每一个epoch的模型
        torch.save({
            "epoch": epoch,
            "model": model_classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": dev_loss,
            "best_acc": best_acc,
        }, os.path.join(args.classifier_dir, f"model_{epoch}.pth.tar"))

        # 早停机制
        if patience_counter >= args.num_patience:
            logging.info(f"early stopping: patience number {args.num_patience} limit reached, so early stopping!")
            break

    # # 自动调参，报道的结果一定是数值而不是张量等其他类型。
    # nni.report_final_result(best_acc)

    logging.info(f"total time cost {((time.time()-start)/60):.4f} minutes!")
    logging.info("========classifier end=========")


if __name__ == '__main__':
    args = get_parser()

    # 1.设定要调参数的默认值。
    params_classifier = {
              "lr_model": args.lr_model,
              "batch_size": args.batch_size,
              "dropout_rate": args.dropout_rate,
              "num_features": args.num_features
              }

    # # 2.获取搜索空间中的超参数
    # optimized_params = nni.get_next_parameter()
    # params_classifier.update(optimized_params)

    # 训练
    train_classifier(args, params_classifier)