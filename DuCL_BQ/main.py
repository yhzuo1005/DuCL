import logging
import nni
import torch
from config import logging_config_fn, get_parser
from stage1 import train_sent_model
from stage2 import train_pair_model
from classifier import train_classifier

if __name__ == '__main__':

    args = get_parser()

    # 日志文件
    logging_config_fn(logging_dir=args.train_log_dir,
                      logging_filename=args.train_log_file,
                      logging_level=logging.INFO,
                      print_on_screen=False)

    logging.info("======logs of parameters======")
    message = '\n'.join([f'{k:<30}: {v}' for k, v in vars(args).items()])
    logging.info(message)

    # 1.设定要调参数的默认值。
    params = {"lr_bert": args.lr_bert,
              "lr_model": args.lr_model,
              "balance": args.balance,
              "batch_size": args.batch_size,
              "dropout_rate": args.dropout_rate,
              "temp_sup": args.temp_sup,
              "temp_unsup": args.temp_unsup,
              "num_features": args.num_features
              }

    # 2.获取搜索空间中的超参数
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)


    # 训练
    train_sent_model(args, params)
    # train_pair_model(args, params)
    # train_classifier(args, params)