#!/usr/bin/env python
"""
    Local test entrance
"""

from __future__ import absolute_import, division, print_function

import ast
import glob
import logging
import os
import sys
import time
import random
import json
import argparse

from train import _train
from inference import model_fn, predict_fn
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--num_train_epochs', type=int, default=4, metavar='E',
                        help='number of total epochs to run (default: 4)')
    parser.add_argument('--train_batch_size', type=int, default=48, metavar='TBS',
                        help='train batch size, can be signle or multiple GPUs (default: 48)')
    parser.add_argument('--eval_batch_size', type=int, default=12, metavar='EBS',
                        help='eval batch size, on single GPU,(default: 12)')
    parser.add_argument('--weight_decay', type=int, default=0, metavar='WD',
                        help='initial weight_decay (default: 0)')
    parser.add_argument('--learning_rate', type=float, default=4e-05, metavar='LR',
                        help='initial learning rate (default: 4e-05)')
    parser.add_argument('--adam_epsilon', type=float, default=1e-08, 
                        help='initial adam_epsilon (default: 1e-08)')
    parser.add_argument('--warmup_steps', type=int, default=0, 
                        help='initial warmup_steps (default: 0)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='max_grad_norm (default: 1.0)')
   
    parser.add_argument('--model_type', type=str, default='bert')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--task_name', type=str, default='binary')
    parser.add_argument('--output_mode', type=str, default='classification')
    parser.add_argument('--max_seq_length', type=int, default=512)
    
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--fp16_opt_level', type=str, default='O1')

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--logging_steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=1000)
    
    parser.add_argument('--reprocess_input_data', type=bool, default=False)
    
    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    
    # parser.add_argument('--hosts', type=str, default=ast.literal_eval(os.environ['SM_HOSTS']))
    # parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    # parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    # parser.add_argument('--output-dir', type=str, default='./outputs')
    # parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    parser.add_argument('--model-dir', type=str, default='/home/mc00/transformers-classification-sm/model')
    parser.add_argument('--data-dir', type=str, default='/home/mc00/transformers-classification-sm/data')
    parser.add_argument('--output-dir', type=str, default='/home/mc00/transformers-classification-sm/outputs')
    
    args= vars(parser.parse_args())
    print(parser.parse_args())
    do_train = False
    if do_train:
        _train(args)
    else:
        with open("input.txt") as f:
            test_string = f.read().strip()
        print("input:", test_string)
        input_data={"txt":test_string}
        model = model_fn(args['model_dir'])
        start_t =time.time()
        print(predict_fn(input_data, model))
        print("time used in inference ", time.time()-start_t)
