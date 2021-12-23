import os
import glob
import json
import torch
import datasets
import pandas as pd
import transformers
import concurrent.futures
from typing import Optional

from datasets import *
from numba import cuda 
from newspaper import Article
from vncorenlp import VnCoreNLP
from transformers import EncoderDecoderModel
from transformers import AutoTokenizer

from model_builder.tokenizer import SummTokenize
from model_builder.ext_model import ExtBertSummPylight

import parameters
args = parameters.get_args()



def listPaths(path):
    """
    Input: Path of folder file 
    Output: list pathfile 
    """
    pathfiles = list()
    for pathfile in glob.glob(path):
        pathfiles.append(pathfile)
    return pathfiles


def read_content(pathfile):
    """
    Input: Path of txt file
    Output: A dictionary has keys 'original' and 'summary'
    """
    with open(pathfile) as f:
        rows  = f.readlines()
        original = ' '.join(''.join(rows[4:]).split('\n'))
        summary = ' '.join(rows[2].split('\n'))
            
    return {'file' : pathfile,
            'original': original, 
            'summary': summary}


def get_dataframe(pathfiles):
    """
    Input: Path of txt file
    Output: DataFrame path, title, summary
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        data = executor.map(read_content, pathfiles)

    # Make blank dataframe
    data_df = list()
    for d in data:
        data_df.append(d)
    data_df = pd.DataFrame(data_df)
    data_df.dropna(inplace=True)
    data_df = data_df.sample(frac=1).reset_index(drop=True)

    return data_df


def crawl_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text


def abs_sum(input, tokenizer, rdrsegmenter, model, device):
    # text = input('Input: ')
    text = rdrsegmenter.tokenize(input)
    text = ' '.join([' '.join(x) for x in text])

    # tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    model.to(device)

    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # print('Summarization: ', output_str)
    cuda.get_current_device().reset()
    return output_str[0]

model_abs = EncoderDecoderModel.from_pretrained(args.checkpoint)
rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g') 
tokenizer_abs = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)


device = 'cpu'
model_ext = ExtBertSummPylight().to(device)
tokenizer_ext = SummTokenize()
def ext_sum(data, tokenizer, model_ext, THRESHOLD = 0.3):

    # print(data)

    (src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask), lis_tgt = \
        tokenizer.tokenizing_ext_input(**data, is_pad=True)
    src_inp_ids = torch.unsqueeze(src_inp_ids, 0).to(device)
    src_tok_type_ids = torch.unsqueeze(src_tok_type_ids, 0).to(device)
    src_lis_cls_pos = torch.unsqueeze(src_lis_cls_pos, 0).to(device)
    src_mask = torch.unsqueeze(src_mask, 0).to(device)
    masked_out_prob = model_ext.predict_step(batch=[src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask, None],
                                         batch_idx=0)
    masked_out_prob = masked_out_prob.reshape(-1)
    src_lis_tok = data.get('src')

    # print('huy')
    res = ''
    for ids in range(len(src_lis_tok)):
        if masked_out_prob[ids] >= THRESHOLD:
            res += ' '.join(src_lis_tok[ids])
    # print(res)
    
    return res.replace('_', ' ')


def convert_to_json(text, rdrsegmenter):
    result = []
    token = rdrsegmenter.tokenize(text)[0]
    temp = []
    for index, i in enumerate(token):
        temp.append(i)
        if i == '.':
            result.append(temp)
            temp = []
        if (index == len(token) - 1) and i != '.':
            result.append(temp)
    return {'src': result}