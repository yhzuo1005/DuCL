#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes, dropout_rate):
        super(Classifier, self).__init__()
        hidden_size = encoder.fc.weight.shape[1]  # 768
        # 768  2
        # self.fc = nn.Linear(hidden_size, num_classes)
        #768,192; 192,2
        # self.classifier = nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(hidden_size, 192), nn.ReLU(),nn.Dropout(dropout_rate), nn.Linear(192, num_classes))  # last is 192

        self.classifier = nn.Sequential(nn.Linear(hidden_size, 192), nn.Tanh(), nn.Linear(192, num_classes))  # last is 192

    def forward(self, features):
        # logits = self.fc(features)
        logits = self.classifier(features)
        probability = F.softmax(logits, dim=-1)
        return logits, probability


# def _init_weight(module):
#     if isinstance(module, nn.Linear):
#         nn.init.xavier_uniform_(module.weight.data)
#         nn.init.constant_(module.bias.data, 0.0)




class BasicBert(BertPreTrainedModel):
    def __init__(self, config):
        super(BasicBert, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)

        # self.apply(_init_weight)




class CLmodel(nn.Module):
    def __init__(self, is_sentence_level, encoder, is_train, pooling=None):
        super(CLmodel, self).__init__()
        self.is_sentence_level = is_sentence_level
        self.pooling = pooling
        self.encoder = encoder.bert
        self.hidden_size = encoder.fc.weight.shape[1]  # 768
        self.is_train = is_train
        self.attention = SoftmaxAttention()
        self.projection = nn.Sequential(
            nn.Linear(4 * self.hidden_size, self.hidden_size),
            nn.ReLU())
        self.pooler = nn.Sequential(nn.Linear(3 * self.hidden_size, self.hidden_size),
                                    encoder.bert.pooler)
        self.head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  nn.ReLU(inplace=True))
        self.fc_sup = encoder.fc  # in:768,out:128
        self.fc_ce = nn.Linear(self.hidden_size, 2)


        self.blockSA = blockSA(self.hidden_size)
        self.layernorm = nn.LayerNorm(self.hidden_size)



    def forward(self, input_ids, attention_mask, token_type_ids):
        if self.is_sentence_level is True:
            out = self.encoder(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
            if self.pooling == None:
                return out
            if self.pooling == 'cls':
                return out.last_hidden_state[:, 0]  # [batch, 768]
            if self.pooling == 'pooler':
                return out.pooler_output  # [batch, 768]
            if self.pooling == 'last-avg':
                last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
                return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            if self.pooling == 'first-last-avg':
                first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
                last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
                first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
                last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
                avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
                return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]
        else:
            input_ids1, input_ids2 = input_ids
            attention_mask1, attention_mask2 = attention_mask
            token_type_ids1, token_type_ids2 = token_type_ids

            feat1 = self.encoder(input_ids=input_ids1,
                                 attention_mask=attention_mask1,
                                 token_type_ids=token_type_ids1
                                 )
            feat2 = self.encoder(input_ids=input_ids2,
                                 attention_mask=attention_mask2,
                                 token_type_ids=token_type_ids2
                                 )
            encoded_premises = feat1[0]
            encoded_hypotheses = feat2[0]

            # # dropout
            # encoded_premises = self.dropout(encoded_premises)
            # encoded_hypotheses = self.dropout(encoded_hypotheses)


            block_size = 3
            seq_len = encoded_premises.size(1)  # 30
            m = seq_len//block_size  # 10 blocks

            premises_blockSA = self.blockSA(encoded_premises) # (bs,block,dim)
            hypotheses_blockSA = self.blockSA(encoded_hypotheses) # (bs,block,dim)


            block_premises_mask = torch.stack([attention_mask1.narrow(1, i, block_size) for i in range(0, attention_mask1.size(1), block_size)], dim=1)  # (bs,block,block_size)
            block_hypotheses_mask = torch.stack([attention_mask2.narrow(1, i, block_size) for i in range(0, attention_mask2.size(1), block_size)], dim=1)  # (bs,block,block_size)

            block_premises_mask_last = torch.where(torch.sum(block_premises_mask, dim=-1) > 0, 1, 0)    # (bs, block)
            block_hypotheses_mask_last = torch.where(torch.sum(block_hypotheses_mask, dim=-1) > 0, 1, 0)  # (bs, block)


            block_premises_att, block_hypotheses_att = self.attention(premises_blockSA, block_premises_mask_last, hypotheses_blockSA, block_hypotheses_mask_last) #(bs,block,dim),(bs,block,dim)

            block_premises_ATT = torch.cat([torch.stack([block_premises_att.select(1, i)]*block_size, dim=1) for i in range(block_premises_att.size(1))], dim=1) #(bs,seq_len,dim)
            block_hypotheses_ATT = torch.cat([torch.stack([block_hypotheses_att.select(1, i)] * block_size, dim=1) for i in range(block_hypotheses_att.size(1))], dim=1) #(bs,seq_len,dim)


            attended_premises_first, attended_hypotheses_first = self.attention(encoded_premises, attention_mask1, encoded_hypotheses, attention_mask2)


            attended_premises = self.layernorm(block_premises_ATT+attended_premises_first)
            attended_hypotheses = self.layernorm(block_hypotheses_ATT+attended_hypotheses_first)


            enhanced_premises = torch.cat([encoded_premises, attended_premises,
                                           encoded_premises - attended_premises,
                                           encoded_premises * attended_premises], dim=-1)
            enhanced_hypotheses = torch.cat([encoded_hypotheses, attended_hypotheses,
                                             encoded_hypotheses - attended_hypotheses,
                                             encoded_hypotheses * attended_hypotheses], dim=-1)
            projected_premises = self.projection(enhanced_premises)
            projected_hypotheses = self.projection(enhanced_hypotheses)
            # pair_embeds = torch.cat([projected_premises, projected_hypotheses, projected_premises - projected_hypotheses, projected_premises * projected_hypotheses], dim=-1)

            pair_embeds = torch.cat([projected_premises, projected_hypotheses, torch.abs(projected_premises - projected_hypotheses)], dim=-1)

            pair_output = self.pooler(pair_embeds)
            if self.is_train:
                feat = self.head(pair_output)
                ce_logits = self.fc_ce(feat)
                ce_probs = F.softmax(ce_logits, dim=-1)
                sup_feat = F.normalize(self.fc_sup(feat), dim=1)
                return ce_logits, ce_probs, sup_feat
            else:
                return pair_output



class customizedModule(nn.Module):
    def __init__(self):
        super(customizedModule, self).__init__()


    def customizedLinear(self, in_dim, out_dim, activation=None, dropout=None):
        cl = nn.Sequential(nn.Linear(in_dim, out_dim))
        nn.init.xavier_uniform(cl[0].weight)
        nn.init.constant(cl[0].bias, 0)

        if activation is not None:
            cl.add_module(str(len(cl)), activation)
        if dropout:
            cl.add_module(str(len(cl)), nn.Dropout(0.3))
        return cl



class s2tSA(customizedModule):
    def __init__(self, hidden_size):
        super(s2tSA, self).__init__()
        self.s2t_W1 = self.customizedLinear(hidden_size, hidden_size, activation=nn.ReLU())
        self.s2t_W = self.customizedLinear(hidden_size, hidden_size)

    def forward(self, x):
        f = self.s2t_W1(x)
        f = F.softmax(self.s2t_W(f), dim=-2)
        s = torch.sum(f*x, dim=-2)
        return s


class blockSA(customizedModule):
    def __init__(self, hidden_size):
        super(blockSA, self).__init__()
        self.hidden_size = hidden_size
        self.s2tSA = s2tSA(hidden_size)
        self.init_SA()
        self.init_blockSA()

    def init_SA(self):
        self.m_W1 = self.customizedLinear(self.hidden_size, self.hidden_size)
        self.m_W2 = self.customizedLinear(self.hidden_size, self.hidden_size)
        self.m_b = nn.Parameter(torch.zeros(self.hidden_size))

        self.m_W1[0].bias.requires_grad = False
        self.m_W2[0].bias.requires_grad = False

        self.c = nn.Parameter(torch.Tensor([5.0]), requires_grad=False)

    def init_blockSA(self):
        self.g_W1 = self.customizedLinear(self.hidden_size, self.hidden_size)
        self.g_W2 = self.customizedLinear(self.hidden_size, self.hidden_size)
        self.g_b = nn.Parameter(torch.zeros(self.hidden_size))

        self.m_W1[0].bias.requires_grad = False
        self.m_W2[0].bias.requires_grad = False

        self.f_W1 = self.customizedLinear(self.hidden_size*3, self.hidden_size, activation=nn.ReLU())
        self.f_W2 = self.customizedLinear(self.hidden_size*3, self.hidden_size)

    def t2tSA(self, x):
        seq_len = x.size(-2)
        x_i = self.m_W1(x).unsqueeze(-2)
        x_j = self.m_W2(x).unsqueeze(-3)

    #     x: (bs, seq_len ,dim)

        f = self.c*F.tanh((x_i+x_j+self.m_b)/self.c)
        f_score = F.softmax(f, dim=-2)
        s = torch.sum(f_score*x.unsqueeze(-2), dim=-2)
        return s



    def forward(self, x):
        r = 3

        x = torch.stack([x.narrow(1, i, r) for i in range(0, x.size(1), r)], dim=1)
        h = self.t2tSA(x)
        v = self.s2tSA(h)

        o = self.t2tSA(v)
        G = F.sigmoid(self.g_W1(o)+self.g_W2(v)+self.g_b)

        e = G*o + (1-G)*v

        # E = torch.cat([torch.stack([e.select(1, i)]*r, dim=1) for i in range(e.size(1))], dim=1)
        # x = x.view(x.size(0), -1, x.size(-1))
        # h = h.view(h.size(0), -1, h.size(-1))
        #
        # fusion = self.f_W1(torch.cat([x, h, E], dim=2))
        # G = F.sigmoid(self.f_W2(torch.cat([x, h, E], dim=2)))
        #
        # u = G*fusion + (1-G)*x

        return e











class SoftmaxAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.

    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """

    def forward(self,
                premise_batch,
                premise_mask,
                hypothesis_batch,
                hypothesis_mask):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.

        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
            attended_hypotheses: The sequences of attention vectors for the
                hypotheses in the input batch.
        """
        # Dot product between premises and hypotheses in each sequence of
        # the batch.
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1)
                                                              .contiguous())

        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2)
                                                        .contiguous(),
                                       premise_mask)

        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch,
                                         prem_hyp_attn,
                                         premise_mask)
        attended_hypotheses = weighted_sum(premise_batch,
                                           hyp_prem_attn,
                                           hypothesis_mask)

        return attended_premises, attended_hypotheses


# Code widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).

    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.

    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)

# Code widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.

    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.

    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask
