# coding: utf-8

from __future__ import absolute_import, division, print_function

import ast
import glob
import logging
import os
import sys
import time
import random
import json

import numpy as np
import torch
import argparse
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import random
from torch.utils.data.distributed import DistributedSampler


from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer, 
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)


from utils import (convert_examples_to_features, InputExample, convert_example_to_feature,
                        output_modes, processors)

import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# globals

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}


JSON_CONTENT_TYPE = 'application/json'
model_type = 'bert'
model_name ='bert-base-uncased'
task_name='binary'
label_map = {'0': 0, '1': 1}
output_mode='classification'

# intialize config and tokenizer, as it is used by both training and inference
config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
config = config_class.from_pretrained(model_name, num_labels=2, finetuning_task=task_name)
tokenizer = tokenizer_class.from_pretrained(model_name)

def model_fn(model_dir):
    logger.info('model_fn')
       
    # initialization

    model = model_class.from_pretrained(model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    return model.to(device)


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


def predict_fn(input_data, model):
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('input_data ' + str(input_data))
    example=InputExample(str(uuid.uuid1()), input_data['txt'], None, '0')
    
    cls_token_at_end=bool(model_type in ['xlnet'])           # xlnet has a cls token at the end
    cls_token=tokenizer.cls_token,
    sep_token=tokenizer.sep_token,
    cls_token_segment_id=2 if model_type in ['xlnet'] else 0
    pad_on_left=bool(model_type in ['xlnet'])               # pad on the left for xlnet
    pad_token_segment_id=4 if model_type in ['xlnet'] else 0
    
    example_row =  (example, label_map, 512, tokenizer, output_mode, cls_token_at_end, cls_token, sep_token, cls_token_segment_id, pad_on_left, pad_token_segment_id, False)
    f = convert_example_to_feature(example_row)

    all_input_ids = torch.tensor([f.input_ids ], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask ], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids ], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id ], dtype=torch.long)

     
    model.eval()
    with torch.no_grad():
        inputs = {'input_ids':      all_input_ids,
                      'attention_mask': all_input_mask,
                      'token_type_ids': all_segment_ids if model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         all_label_ids }
        outputs = model(**inputs)
        _, logits = outputs[:2]

    preds = logits.detach().cpu().numpy()
    print(preds)         
    preds = np.argmax(preds, axis=1)
    print(preds[0])
    return str(preds[0])
