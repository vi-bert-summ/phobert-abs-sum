import torch
from torch.nn.functional import pad

from model_builder.config import *


def get_cls_embed(tok_embed, cls_pos):
    """
        Hàm này sẽ lấy ra tok_embed của các token [CLS]
       :param tok_embed: có shape (n_batch * n_seq * n_embed)
       :param cls_pos: vị trí của các [CLS] token có shape (n_batch * số lượng câu trong input của batch đó )
    """
    filter_cls_pos = [[int(pos) for pos in single_cls_pos if pos > 0] for single_cls_pos in cls_pos]
    return [each_batch_tok_embed[filter_cls_pos[i]] for i, each_batch_tok_embed in enumerate(tok_embed)]


def padding_seq(tok_seq):
    """
    tok_seq: (n_seq * n_embed)
    Hàm này sẽ padding một sequence sao cho n_seq = MAX_TOK_LEN
    """
    n_seq, n_embed = tok_seq.shape
    device = torch.device('cpu')
    # if torch.cuda.is_available():
        # device = tok_seq.get_device()
    padded_tok_seq = pad(input=tok_seq, pad=(0, 0, 0, MAX_SEQ_LENGTH - n_seq), mode='constant', value=0)
    mask = torch.zeros(MAX_SEQ_LENGTH, device=device, dtype=torch.long)
    mask[:n_seq] = 1
    return padded_tok_seq, mask


def padding_and_stack_cls(tok_embed):
    """
    Hàm này cần padding tok_embed sao cho n_seq = MAX_SEQ_LEN
    :param tok_embed( n_batch * n_seq * n_embed)
    """
    padding_seq_res = [padding_seq(each_batch_tok_embed) for each_batch_tok_embed in tok_embed]
    padded_tok_embed = torch.stack([single_padding_seq[0] for single_padding_seq in padding_seq_res])
    mask = torch.stack([single_padding_seq[1] for single_padding_seq in padding_seq_res])
    return padded_tok_embed, mask


DEFAULT_MASK_MATRIX = torch.triu(torch.ones(size=(MAX_SEQ_LENGTH, MAX_SEQ_LENGTH)), diagonal=1).T