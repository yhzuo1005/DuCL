import random
from tqdm import tqdm

# 将三个集合的句子合为一个集合的句子。
def read_data(train_file, dev_file, test_file, tgt_file):
    with open(train_file, "r", encoding="utf8") as read_file_train, open(dev_file, "r", encoding="utf8") as read_file_dev, \
            open(test_file, "r", encoding="utf8") as read_file_test, open(tgt_file, "w", encoding="utf8") as write_file:
        # 写入标题行
        write_file.writelines("sent"+"\n")
        for idx_line, line in enumerate(read_file_train):
            line_new = line.strip().split("\t")
            sent1, sent2, label = line_new[0], line_new[1], int(line_new[2])
            write_file.writelines(sent1 + "\n")
            write_file.writelines(sent2 + "\n")

        for idx_line, line in enumerate(read_file_dev):
            line_new = line.strip().split("\t")
            sent1, sent2, label = line_new[0], line_new[1], int(line_new[2])
            write_file.writelines(sent1 + "\n")
            write_file.writelines(sent2 + "\n")

        for idx_line, line in enumerate(read_file_test):
            line_new = line.strip().split("\t")
            sent1, sent2, label = line_new[0], line_new[1], int(line_new[2])
            if idx_line == 10000 - 1:
                write_file.writelines(sent1 + "\n")
                write_file.writelines(sent2)
            else:
                write_file.writelines(sent1 + "\n")
                write_file.writelines(sent2 + "\n")


# # 将句子对转换为单个句子
# read_data(train_file="./dataset/LCQMC/LCQMC.train.data", dev_file="./dataset/LCQMC/LCQMC.valid.data", test_file="./dataset/LCQMC/LCQMC.test.data", tgt_file="./dataset/LCQMC/LCQMC.sent_all.txt")



# 去除txt文件中的重复行
def drop_duplicate(src_file, tgt_sent_file, tgt_idx_file):
    list_sent, list_idx = [], []
    with open(src_file, "r", encoding="utf8") as read_file, open(tgt_sent_file, "w", encoding="utf8") as write_sent_file, open(tgt_idx_file, "w", encoding="utf8") as write_idx_file:
        # sent_all是所有句子构成的列表
        sent_all = read_file.readlines()
        for idx, sent in enumerate(tqdm(sent_all)):
            # 标题行是第0行数据
            if idx == 0:
                print("This is a header line!")
            elif sent not in list_sent:
                list_sent.append(sent)
                list_idx.append(idx)
        # 因为第一行是标题行，所以-1。
        print(f"原始的数据量为{len(sent_all)-1}")
        print(f"去重以后的列表长度为{len(list_sent)}.")

        for sent in list_sent:
            write_sent_file.writelines(sent)

        for idx in list_idx:
            write_idx_file.writelines(str(idx)+"\n")


# drop_duplicate(src_file="./dataset/LCQMC/LCQMC.sent_all.txt", tgt_sent_file="./dataset/LCQMC/LCQMC.sent_unique.txt",
#                tgt_idx_file="./dataset/LCQMC/LCQMC.sent_unique_idx.txt")


def merge_fn(idx_file, sent_file, merge_file):
    with open(idx_file, "r", encoding="utf8") as read_idx, open(sent_file, "r", encoding="utf8") as read_sent, open(merge_file, "w", encoding="utf8") as write_idx_sent:

        list_idx = read_idx.readlines()
        list_sent = read_sent.readlines()

        num_idx = len(list_idx)
        num_sent = len(list_sent)
        assert num_idx == num_sent
        print(f"索引数量=句子数量;索引数量为{num_idx},句子数量为{num_sent}.")

        for idx, sent in zip(list_idx, list_sent):
            idx = idx.strip("\n")
            write_idx_sent.writelines(idx+"\t"+sent)


# merge_fn(idx_file="./dataset/LCQMC/LCQMC.sent_unique_idx.txt", sent_file="./dataset/LCQMC/LCQMC.sent_unique.txt", merge_file="./dataset/LCQMC/LCQMC.sent_unique_merge.txt")




# 抽取一定数量的数据，比如10000条数据。
def extract_data(src_file, tgt_file_sent_id, tgt_file_sent, num_sent=10000):
    with open(src_file, "r", encoding="utf8") as read_file, open(tgt_file_sent_id, "w", encoding="utf8") as write_sent_id, open(tgt_file_sent, "w", encoding="utf8") as write_sent:
        list_sent_id = read_file.readlines()
        selected_sent_id = random.sample(list_sent_id, num_sent)
        for idx, sent in enumerate(tqdm(selected_sent_id)):
            write_sent_id.writelines(sent)
            id, sentence = sent.split("\t")

            write_sent.writelines(sentence)

extract_data(src_file="./dataset/LCQMC/LCQMC.sent_unique_merge.txt", tgt_file_sent_id="./dataset/LCQMC/LCQMC.sent_id_extract.txt",
             tgt_file_sent="./dataset/LCQMC/LCQMC.sent_extract.txt", num_sent=40000)