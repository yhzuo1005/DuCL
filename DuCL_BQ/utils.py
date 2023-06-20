import nni
import logging
import os
import time
from scipy.stats import spearmanr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import Metrics
from model import CLmodel, BasicBert


# 无监督的损失函数
def simcse_unsup_loss(y_pred, device, temp=0.05):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]
    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = y_true + 1 - y_true % 2 * 2

    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    # [batch_size * 2, 1, 768] * [1, batch_size * 2, 768] = [batch_size * 2, batch_size * 2]
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)

    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    sim = sim / temp  # 相似度矩阵除以温度系数

    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    loss_mean = torch.mean(loss)
    return loss_mean



def train_unsup(model_unsup,unsup_dataloader, sup_dataloader_dev, simcse_unsup_loss,temp_unsup, optimizer, device, sent_model_dir):
    # 训练模式
    model_unsup.train()
    best_corrcoef = 0.0

    for batch_idx, batch in enumerate(tqdm(unsup_dataloader)):
        batch_size = batch.get("input_ids").shape[0]
        input_ids = batch.get("input_ids").view(batch_size*2, -1).to(device)
        attention_mask = batch.get("attention_mask").view(batch_size*2, -1).to(device)
        token_type_ids = batch.get("token_type_ids").view(batch_size*2, -1).to(device)

        # 优化器梯度清零
        optimizer.zero_grad()

        # output = "cls"
        output = model_unsup(input_ids, attention_mask, token_type_ids).last_hidden_state[:, 0]

        # 计算损失值
        loss = simcse_unsup_loss(output, device, temp_unsup)

        loss.backward()
        optimizer.step()

        # 每5个batch去验证集查看一下指标变化情况
        if batch_idx % 5 == 0:
            logging.info(f"batch is {batch_idx}, loss is {loss.item():.4f}")
            corrcoef = evaluation_dev(model_unsup, sup_dataloader_dev, device)

            # 在验证以后，需要重启训练模式
            model_unsup.train()

            # 保存最好的模型
            if best_corrcoef < corrcoef:
                best_corrcoef = corrcoef
                torch.save({
                    "batch_idx": batch_idx,
                    "model": model_unsup.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_corrcoef": best_corrcoef,
                    "loss": loss.item()
                }, os.path.join(sent_model_dir, "best.pth.tar"))
                logging.info(f"the better corrcoef is {best_corrcoef:.4f} in batch {batch_idx}, save model in {sent_model_dir}.")



    return best_corrcoef





def evaluation_dev(model_unsup, sup_dataloader_dev, device):
    # 进入模型验证模式
    model_unsup.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])

    with torch.no_grad():
        for source, target, label in tqdm(sup_dataloader_dev):
            source_input_ids = source.get("input_ids").squeeze(1).to(device)
            source_attention_mask = source.get("attention_mask").squeeze(1).to(device)
            source_token_type_ids = source.get("token_type_ids").squeeze(1).to(device)

            # source_cls
            source_pred = model_unsup(source_input_ids, source_attention_mask, source_token_type_ids).last_hidden_state[:, 0]

            target_input_ids = target.get("input_ids").squeeze(1).to(device)
            target_attention_mask = target.get("attention_mask").squeeze(1).to(device)
            target_token_type_ids = target.get("token_type_ids").squeeze(1).to(device)

            # target_cls
            target_pred = model_unsup(target_input_ids, target_attention_mask, target_token_type_ids).last_hidden_state[:, 0]

            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    # 计算整个验证集的corrcoef
    corrcoef = spearmanr(label_array, sim_tensor.cpu().numpy()).correlation
    return corrcoef


def test_fn(model_unsup, sup_dataloader, device):
    # 进入模型验证模式
    model_unsup.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in tqdm(sup_dataloader):
            source_input_ids = source.get("input_ids").squeeze(1).to(device)
            source_attention_mask = source.get("attention_mask").squeeze(1).to(device)
            source_token_type_ids = source.get("token_type_ids").squeeze(1).to(device)
            # cls
            source_pred = model_unsup(source_input_ids, source_attention_mask, source_token_type_ids).last_hidden_state[:, 0]

            target_input_ids = target.get("input_ids").squeeze(1).to(device)
            target_attention_mask = target.get("attention_mask").squeeze(1).to(device)
            target_token_type_ids = target.get("token_type_ids").squeeze(1).to(device)
            # cls
            target_pred = model_unsup(target_input_ids, target_attention_mask, target_token_type_ids).last_hidden_state[:, 0]

            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    # 测试模型在train/dev/test集合上的表现
    corrcoef = spearmanr(label_array, sim_tensor.cpu().numpy()).correlation
    return corrcoef



def test_unsup(pretrained_model_path,num_features, dropout_rate,sent_model_dir, device, sup_dataloader_train, sup_dataloader_dev, sup_dataloader_test):

    # 初始化无监督模型
    model_unsup = CLmodel(is_sentence_level=True,
                          encoder=BasicBert.from_pretrained(
                          pretrained_model_path,
                          num_labels=num_features,
                          attention_probs_dropout_prob=dropout_rate,
                          hidden_dropout_prob=dropout_rate,
                          output_attentions=False,
                          output_hidden_states=False
                          ),
                          is_train=False,
                          pooling=None
    ).to(device=device)

    logging.info(f"loading the model_unsup from file {sent_model_dir}/best.pth.tar")

    # 加载表现最佳的无监督模型
    checkpoint_unsup_model = torch.load(os.path.join(sent_model_dir, "best.pth.tar"))
    model_unsup.load_state_dict(checkpoint_unsup_model["model"])

    corrcoef_train = test_fn(model_unsup, sup_dataloader_train, device)
    corrcoef_dev = test_fn(model_unsup, sup_dataloader_dev, device)
    corrcoef_test = test_fn(model_unsup, sup_dataloader_test, device)

    # 测试模型在train,dev,test中的性能
    return corrcoef_train, corrcoef_dev, corrcoef_test



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.05, contrast_mode='all', base_temperature=0.05):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model.

        Args:
            features: hidden vector of shape [bsz, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_feature = features
        contrast_count = 1
        anchor_feature = features
        anchor_count = 1

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def train_sup(model_sup, sup_dataloader_train, optimizer, criterion_ce, criterion_sup, device, balance):
    # 训练模式
    model_sup.train()

    epoch_start = time.time()
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

        ce_logits, ce_probs, sup_feat = model_sup(input_ids, attention_mask, token_type_ids)

        optimizer.zero_grad()

        loss_ce = criterion_ce(ce_logits, label.to(device))
        loss_pair = criterion_sup(sup_feat, label.to(device))

        loss = loss_ce + balance*loss_pair
        loss.backward()

        optimizer.step()

        batch_time_avg += time.time()-batch_start
        running_loss += loss.item()

        description = f"Avg batch time is {batch_time_avg/(batch_idx+1):.4f}s, loss is {running_loss/(batch_idx+1):.4f}."
        tqdm(sup_dataloader_train).set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss/len(sup_dataloader_train)

    return epoch_time, epoch_loss


def dev_sup(model_sup, sup_dataloader_dev, criterion_ce, criterion_sup, device, balance):
    model_sup.eval()

    start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
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

            ce_logits, ce_probs, sup_feat = model_sup(input_ids, attention_mask, token_type_ids)

            loss_ce = criterion_ce(ce_logits, label.to(device))
            loss_pair = criterion_sup(sup_feat, label.to(device))
            # loss_pair = criterion_sup(sup_feat.detach(), label.to(device))

            loss = loss_ce + balance * loss_pair
            running_loss += loss.item()

            trues.extend(label.numpy())
            preds.extend(ce_probs[:, 1].detach().cpu().numpy())

            batch_time_avg += time.time() - batch_start
            description = f"Avg batch time is {batch_time_avg / (batch_idx + 1):.4f}s, loss is {running_loss / (batch_idx + 1):.4f}."
            tqdm(sup_dataloader_dev).set_description(description)
    trues = np.array(trues)
    preds = np.array(preds)

    predict = [1 if x >= 0.5 else 0 for x in preds]
    precision, recall, f1, acc = Metrics(trues, predict)

    dev_time = time.time() - start
    dev_loss = running_loss / len(sup_dataloader_dev)

    return dev_time, dev_loss, precision, recall, f1, acc