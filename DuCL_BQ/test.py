import logging
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from config import get_parser, init_seed, logging_config_fn, Metrics
from data_process import load_sup_data, supDataset
from model import CLmodel, Classifier, BasicBert


def test_sup(model_sup, model_classifier, test_dataloader, device):
    model_sup.eval()
    model_classifier.eval()

    start = time.time()
    batch_time = 0.0
    batch_time_avg = 0.0
    trues, preds = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
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

            trues.extend(label.numpy())
            preds.extend(probs[:, 1].detach().cpu().numpy())

            batch_time_avg += time.time()-batch_start
            description = f"Avg batch time is {(batch_time_avg/(batch_idx+1)):.4f}s."
            tqdm(test_dataloader).set_description(description)

            batch_time = time.time()-batch_start

    trues = np.array(trues)
    preds = np.array(preds)

    predict = [1 if x >= 0.5 else 0 for x in preds]
    precision, recall, f1, acc = Metrics(trues, predict)

    batch_time /= len(test_dataloader)
    total_time = time.time()-start

    return batch_time, total_time, precision, recall, f1, acc




def test_model():
    args = get_parser()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    init_seed(args.seed)

    # 日志文件
    logging_config_fn(logging_dir=args.test_log_dir,
                      logging_filename=args.test_log_file,
                      logging_level=logging.INFO,
                      print_on_screen=False)

    logging.info("======logs of parameters======")
    message = '\n'.join([f'{k:<30}: {v}' for k, v in vars(args).items()])
    logging.info(message)


    logging.info("========test model begin=========")
    logging.info(f"Dataset is {args.data}")

    logging.info("Initializing the model....")
    model_sup = CLmodel(is_sentence_level=False,
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

    model_classifier = Classifier(
            encoder=BasicBert.from_pretrained(
            args.pretrained_model_path,  # Use the 12-layer BERT model
            num_labels=args.label_size,
            output_attentions=False,
            output_hidden_states=False,
        ),
        num_classes=args.label_size,
        dropout_rate=args.dropout_rate
    ).to(args.device)


    logging.info(f"loading model_sup from ckpt model_best.pth.tar.")
    ckpt_model_sup = torch.load(os.path.join(args.pair_model_dir, "best.pth.tar"))
    model_sup.load_state_dict(ckpt_model_sup["model"])

    logging.info(f"loading model_classifier from {args.classifier_dir}/best.pth.tar.")
    ckpt_classifier = torch.load(os.path.join(args.classifier_dir, "best.pth.tar"))
    model_classifier.load_state_dict(ckpt_classifier["model"])

    logging.info(f"loading the test data from {args.test_data}.")

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    sup_test = load_sup_data(args.test_data)
    test_dataset = supDataset(data=sup_test, tokenizer=tokenizer, max_len=args.max_len)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    batch_time, total_time, precision, recall, f1, acc = test_sup(model_sup, model_classifier, test_dataloader, args.device)
    logging.info(f"Avg one batch time is {batch_time:.4f}s, total test time is {total_time:.4f}s.")
    logging.info(f"===metrics: precision is {precision:.4f}, recall is {recall:.4f}, f1 is {f1:.4f}, acc is {acc:.4f}===")
    logging.info(f"========balance is {args.balance}=======")
    logging.info("========test model end=========")


# 找到最佳参数以后，将最佳参数赋值给原来的默认参数值，然后运行此文件即可。
# 只需要在这个文件中记录下模型参数值即可，之前文件中不必记录，因为之前文件中的参数值是原先的默认值。
if __name__ == '__main__':
    test_model()