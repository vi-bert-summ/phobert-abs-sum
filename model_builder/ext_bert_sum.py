import torch

from torch import nn
from torch.nn import Sequential

from pytorch_lightning import LightningModule
from transformers import BertModel, get_linear_schedule_with_warmup, AdamW

from torch.nn.functional import mse_loss

from model_builder.config import *
from model_builder.utils import get_cls_embed, padding_and_stack_cls
from torch.nn import CrossEntropyLoss


class ExtBertSumm(nn.Module):
    def __init__(self):
        super(ExtBertSumm, self).__init__()
        self.phase1_bert = PhoBert(large=False, temp_dir=CACHED_MODEL_PATH, is_freeze=False)
        self.phase2_decoder = ExtDecoder()
        self.sent_classifier = Sequential(
            nn.Linear(MODEL_DIM, 1)
        )

    def forward(self, src_ids, src_pad_mask, src_token_type, src_cls_pos):
        """
        :param src_ids: embeding token của src  (batch_size * seq_len)
        :param src_pad_mask: đánh dấu padding của src (để attention không tính đến nó nữa) (batch_size * seq_len)
        :param src_token_type: đánh dấu đoạn A và B của câu src (batch_size * seq_len)
        :param src_cls_pos: vị trí của các token cls trong src seq (batch_size * num_cls)
        :return:
        """
        # n_batch * n_tokens * n_embed_dim
        embed_phase1 = self.phase1_bert(input_ids=src_ids,
                                        token_type_ids=src_token_type,
                                        attention_mask=src_pad_mask)

        cls_embed = get_cls_embed(tok_embed=embed_phase1, cls_pos=src_cls_pos)  # n_batch * n_cls * n_embed
        padded_cls_embed, pad_mask = padding_and_stack_cls(cls_embed)  # n_batch * MAX_SEQ_LENGTH * n_embed

        out = self.phase2_decoder(src_ids=padded_cls_embed, src_mask=pad_mask)

        logits = self.sent_classifier(out)
        return logits


class ExtDecoder(nn.Module):
    def __init__(self, n_head=8, n_encoder_block=6):
        super(ExtDecoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(batch_first=True, d_model=MODEL_DIM, nhead=n_head, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_block)

    def forward(self, src_ids, src_mask):
        """
        :param src_ids: (n_batch * MAX_SEQ_LENGTH * n_embed) Embed Cls của BERT phase 1
        :param src_mask: (n_batch * MAX_SEQ_LENGTH * n_embed) Mask của đầu vào
        """
        out = self.transformer_encoder(src=src_ids, src_key_padding_mask=src_mask)
        return out


class PhoBert(nn.Module):
    """
    Đầu vào: (batch_size, sequence_len)
    sequence_len <= 258.
    """

    def __init__(self, large, temp_dir, is_freeze):
        super(PhoBert, self).__init__()
        # Lựa chọn mô hình BERT-large hoặc BERT-base
        if large:
            self.model = BertModel.from_pretrained(BERT_LARGE_MODEL, cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained(BERT_BASE_MODEL, cache_dir=temp_dir)

        self.config = self.model.config
        self.is_freeze = is_freeze

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        :param input_ids:  embed đầu vào (n_batch * n_seq)
        :param token_type_ids: đánh dấu tok nào thuộc đoạn nào (n_batch * n_seq)
        :param attention_mask: đánh dấu đâu là padding token (n_batch * n_seq)
        :param is_freeze: có train weight của mô hình hay không
        :return:
        """
        # print(input_ids, token_type_ids, attention_mask)
        if not self.is_freeze:
            _ = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask)
        else:
            self.model.eval()
            with torch.no_grad():
                _ = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask)

        return _[0]