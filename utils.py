import os
import glob
import datasets
import pandas as pd
import transformers
import concurrent.futures
from typing import Optional

from datasets import *
from newspaper import Article
from vncorenlp import VnCoreNLP
from transformers import EncoderDecoderModel
from transformers import AutoTokenizer


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


def bertsum(input, tokenizer, rdrsegmenter, device, checkpoint_file):
    # text = input('Input: ')
    text = rdrsegmenter.tokenize(input)
    text = ' '.join([' '.join(x) for x in text])

    # tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    model = EncoderDecoderModel.from_pretrained(checkpoint_file)
    model.to(device)

    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # print('Summarization: ', output_str)

    return output_str[0]
