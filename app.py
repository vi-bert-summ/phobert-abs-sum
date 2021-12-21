import os
import nltk
import torch
import urllib.request
import streamlit as st

import utils

from newspaper import Article
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
from transformers import EncoderDecoderModel


args = parameters.get_args()


def main():
    st.markdown("<h1 style='text-align: center;'>Capstone Project Summary✏️</h1>", unsafe_allow_html=True)

    # Download model
    if not os.path.exists('checkpoints/mobilebert_ext.pt'):
        # download_model()
        pass

    # Load model
    # model = load_model()

    # Input
    input_type = st.radio("Input Type: ", ["URL", "Raw Text"])
    st.markdown("<h3 style='text-align: center;'>Input</h3>", unsafe_allow_html=True)

    if input_type == "Raw Text":
        text = st.text_input("")
    else:
        url = st.text_input("", "https://www.cnn.com/2020/05/29/tech/facebook-violence-trump/index.html")
        st.markdown(f"[*Read Original News*]({url})")
        text = crawl_url(url)
        summary = text

    # Summarize
    sum_level = st.radio("Output Length: ", ["Short", "Medium"])
    max_length = 3 if sum_level == "Short" else 5
    result_fp = 'results/summary.txt'
    # summary = summarize(input_fp, result_fp, model, max_length=max_length)
    st.markdown("<h3 style='text-align: center;'>Summary</h3>", unsafe_allow_html=True)
    st.markdown(f"<p align='justify'>{summary}</p>", unsafe_allow_html=True)

@st.cache(suppress_st_warning=True)
def load_model():
    model = EncoderDecoderModel.from_pretrained(args.save_model)
    model.to(args.device)
    return model


def crawl_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text


if __name__ == "__main__":
    main()


