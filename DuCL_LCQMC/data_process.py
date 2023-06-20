#coding:utf-8
from torch.utils.data import Dataset


def load_unsup_data(train_path, dev_path, test_path):
    all_sent1, all_sent2, all_sent, all_label = [], [], [], []
    for dataset_type in [train_path, dev_path, test_path]:
        with open(dataset_type, "r", encoding="utf8") as readfile:
            for line in readfile:
                line_new = line.strip().split("\t")
                all_sent1.append(line_new[0])
                all_sent2.append(line_new[1])
                all_label.append(int(line_new[2]))
    all_sent.extend(all_sent1)
    all_sent.extend(all_sent2)

    return all_sent1, all_sent2, all_sent, all_label



def load_sup_data(data_path):
    data = []
    with open(data_path, "r", encoding="utf8") as readfile:
        for line in readfile:
            line_new = line.strip().split("\t")
            data.append((line_new[0], line_new[1], int(line_new[2])))
    return data

def load_extract_data(data_path):
    data = []
    with open(data_path, "r", encoding="utf8") as read_file:
        for line in read_file:
            line_new = line.strip()
            data.append(line_new)
    return data






# 自定义无监督数据集的dataset
class unsupDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text2id(self, text):
        return self.tokenizer([text, text], padding="max_length", max_length=self.max_len, truncation=True,
                       return_tensors="pt")

    def __getitem__(self, item):
        sent = self.data[item]
        tokens = self.text2id(sent)
        return tokens

# 自定义有监督数据集的dataset
class supDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text2id(self, text):
        return self.tokenizer(text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")

    def __getitem__(self, item):
        sent = self.data[item]
        token0 = self.text2id(sent[0])
        token1 = self.text2id(sent[1])
        label = sent[2]
        return token0, token1, label



