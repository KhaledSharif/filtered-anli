#!/bin/bash

. setup.sh
python     src/nli/training.py     --single_gpu     --model_class_name "bert-base"     --max_length 160     --gradient_accumulation_steps 1     --per_gpu_train_batch_size 16     --per_gpu_eval_batch_size 16     --save_prediction     --train_data anli_r1_train:     --train_weights 1     --eval_data anli_r1_dev:,anli_r2_dev:,anli_r3_dev:     --eval_frequency 1000     --epochs 40     --experiment_name "bert-anli_r1" > out.txt
