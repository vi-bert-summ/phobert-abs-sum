import os
import glob
import datasets
import pandas as pd
import transformers
import concurrent.futures
from typing import Optional

import utils
import parameters

from datasets import *
from vncorenlp import VnCoreNLP
from transformers import EncoderDecoderModel
from transformers import RobertaTokenizerFast, AutoTokenizer
from seq2seq_trainer import Seq2SeqTrainer
from transformers import TrainingArguments
from dataclasses import dataclass, field

args = parameters.get_args()


if __name__ == '__main__':
    text = input('Input: ')
    rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g') 

    text = rdrsegmenter.tokenize(text)
    text = ' '.join([' '.join(x) for x in text])

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    model = EncoderDecoderModel.from_pretrained(args.save_model)
    model.to(args.device)

    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    print('Summarization: ', output_str)