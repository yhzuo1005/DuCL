#!/bin/bash
source /mnt/public/home/s-zuoyh1/software/anaconda3531/envs/torch1/bin/ activate
/mnt/public/home/s-zuoyh1/software/anaconda3531/envs/torch1/bin/python /mnt/public/home/s-zuoyh1/DuCL_BQ/stage1.py --balance 0.2 --batch_size 64 --lr_bert 5e-5 --dropout_rate 0.3 --temp_unsup 0.06
/mnt/public/home/s-zuoyh1/software/anaconda3531/envs/torch1/bin/python /mnt/public/home/s-zuoyh1/DuCL_BQ/stage2.py --balance 0.2 --batch_size 64 --lr_bert 5e-5 --dropout_rate 0.3 --temp_unsup 0.06
/mnt/public/home/s-zuoyh1/software/anaconda3531/envs/torch1/bin/python /mnt/public/home/s-zuoyh1/DuCL_BQ/classifier.py --balance 0.2 --batch_size 64 --lr_bert 5e-5 --dropout_rate 0.3 --temp_unsup 0.06
/mnt/public/home/s-zuoyh1/software/anaconda3531/envs/torch1/bin/python /mnt/public/home/s-zuoyh1/DuCL_BQ/test.py --balance 0.2 --batch_size 64 --lr_bert 5e-5 --dropout_rate 0.3 --temp_unsup 0.06
