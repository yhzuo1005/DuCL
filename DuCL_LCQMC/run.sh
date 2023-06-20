#!/bin/bash
source /mnt/public/home/s-zuoyh1/software/anaconda3531/envs/torch1/bin/ activate
/mnt/public/home/s-zuoyh1/software/anaconda3531/envs/torch1/bin/python /mnt/public/home/s-zuoyh1/DuCL_LCQMC/stage1.py --balance 0.9 --batch_size 32 --lr_bert 1e-5 --dropout_rate 0.3 --temp_unsup 0.04
/mnt/public/home/s-zuoyh1/software/anaconda3531/envs/torch1/bin/python /mnt/public/home/s-zuoyh1/DuCL_LCQMC/stage2.py --balance 0.9 --batch_size 32 --lr_bert 1e-5 --dropout_rate 0.3 --temp_unsup 0.04
/mnt/public/home/s-zuoyh1/software/anaconda3531/envs/torch1/bin/python /mnt/public/home/s-zuoyh1/DuCL_LCQMC/classifier.py --balance 0.9 --batch_size 32 --lr_bert 1e-5 --dropout_rate 0.3 --temp_unsup 0.04
/mnt/public/home/s-zuoyh1/software/anaconda3531/envs/torch1/bin/python /mnt/public/home/s-zuoyh1/DuCL_LCQMC/test.py --balance 0.9 --batch_size 32 --lr_bert 1e-5 --dropout_rate 0.3 --temp_unsup 0.04