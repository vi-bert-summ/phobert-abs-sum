import os
import numpy as np
import nltk
from tensorflow.python import util
import torch
import urllib.request
import streamlit as st

import utils

from newspaper import Article
import os
import time
import glob
import datasets
import pandas as pd
import transformers
import concurrent.futures
from typing import Optional

import utils
import parameters

# from datasets import *
from vncorenlp import VnCoreNLP
from transformers import EncoderDecoderModel
from transformers import AutoTokenizer

args = parameters.get_args()



def main():
    start = time.time()
    st.markdown(
        """
        <style>
        .container {
            display: inline-block;
        }
        .logo-img {
            float: left;
            position: fixed;
            top: 20%;
            left: 5%
        }
        .logo-img-2 {
            float: right;
            position: fixed;
            left: 75%;
            top: 20%
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="container">
            <img class="logo-img" src="https://img1.picmix.com/output/stamp/normal/3/7/8/5/1395873_5d4a6.gif">
        </div>
        <div class="container">
            <img class="logo-img-2" src="https://img1.picmix.com/output/stamp/normal/3/7/8/5/1395873_5d4a6.gif">
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<h1 style='text-align: center;'>Tóm Tắt Văn Bản Tiếng Việt</h1>", unsafe_allow_html=True)

    # Load model
    # model = load_model()
    # Input
    input_type = st.radio("Định dạng đầu vào: ", ["URL", "Đoạn văn"])
    st.markdown("<h3 style='text-align: center;'>Đầu vào</h3>", unsafe_allow_html=True)

    if input_type == "Đoạn văn":
        text = st.text_input("")
    else:
        url = st.text_input("", "https://dantri.com.vn/the-thao/hlv-thai-lan-chung-toi-da-co-mot-ket-qua-hoan-hao-20211223213332914.htm")
        st.markdown(f"[*Nguồn*]({url})")
        text = utils.crawl_url(url)

    text = text.replace('_', ' ')
    # Summarize
    sum_level = st.radio("Kiểu tóm tắt: ", ["Trích rút (Extractive)", "Tóm lược (Abstractive)"])
    result = st.button('Tóm tắt')
    summary = ''
    time_consuming = ''
    if result: 
        if sum_level == 'Tóm lược (Abstractive)':
            summary = utils.abs_sum(text, utils.tokenizer_abs, utils.rdrsegmenter, utils.model_abs, args.device)
            time_consuming = int(time.time() - start)
            time_consuming = str(time_consuming) + 's'
        else:
            summary = utils.ext_sum(utils.convert_to_json(text, utils.rdrsegmenter), utils.tokenizer_ext,utils.model_ext)

    summary = summary.replace('_', ' ')
    # max_length = 3 if sum_level == "Short" else 5
    # result_fp = 'results/summary.txt'
    # summary = summarize(input_fp, result_fp, model, max_length=max_length)
    st.markdown("<h3 style='text-align: center;'>Nội dung gốc</h3>", unsafe_allow_html=True)
    st.markdown(f"<p align='justify'>{text}</p>", unsafe_allow_html=True)
    if result: 
        st.markdown("<h3 style='text-align: center;'>Văn bản tóm tắt {}</h3>".format(time_consuming), unsafe_allow_html=True)
        st.markdown(f"<p align='justify'>{summary}</p>", unsafe_allow_html=True)

@st.cache(suppress_st_warning=True)
def load_model():
    model = EncoderDecoderModel.from_pretrained(args.checkpoint)
    model.to(args.device)
    return model


def crawl_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text


if __name__ == "__main__":
    main()


