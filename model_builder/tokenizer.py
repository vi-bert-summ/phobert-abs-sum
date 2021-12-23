import torch

from transformers import AutoTokenizer

from model_builder.config import MAX_SEQ_LENGTH, CACHED_MODEL_PATH, BERT_LARGE_MODEL, BERT_BASE_MODEL



class SummTokenize:
    def __init__(self, is_large=False):
        if is_large:
            self.phobert_tokenizer = AutoTokenizer.from_pretrained(BERT_LARGE_MODEL,
                                                                   cache_dir=CACHED_MODEL_PATH)
        else:
            self.phobert_tokenizer = AutoTokenizer.from_pretrained(BERT_BASE_MODEL,
                                                                   cache_dir=CACHED_MODEL_PATH)
        self.cls_tok_id = self.phobert_tokenizer.convert_tokens_to_ids(self.phobert_tokenizer.cls_token)
        self.max_seq_length = MAX_SEQ_LENGTH

    def combine_tokenize_result(self, lis_tok_res):
        """
        :param lis_tok_res: Kết quả embedding của mỗi câu (dict bao gồm các key input_ids, attention_mask, ... )
        :return: all_input_ids (n_tokens), token_type_ids (n_tokens), lis_cls_pos (n_tokens)
        """
        all_input_ids = torch.cat([torch.tensor(tok_res.get('input_ids')) for tok_res in lis_tok_res], dim=0)
        token_type_ids = torch.tensor([torch.tensor(i % 2) for i, tok_res in enumerate(lis_tok_res) for _ in
                                       range(len(tok_res.get('input_ids')))], dtype=torch.long)
        lis_cls_pos = torch.tensor(
            [i for i, inp_ids in enumerate(all_input_ids) if inp_ids == self.cls_tok_id])

        return all_input_ids, token_type_ids, lis_cls_pos

    def tokenize_formatted_list(self, inp_list):
        """
        :param inp_list: Một xâu bất kỹ đã được segmentation (n_sent * n_token)
        :return: all_input_ids (n_tokens), token_type_ids (n_tokens), lis_cls_pos (n_tokens)
        """
        encoded_seq = [self.phobert_tokenizer(' '.join(sent)) for sent in inp_list]
        return self.combine_tokenize_result(encoded_seq)

    def padding_seq(self, input_ids, token_type_ids, lis_cls_pos):
        """
        :param input_ids: list các ids
        :param token_type_ids: loại token (dùng để phân đoạn câu A / B)
        :param lis_cls_pos: list vị trí của các token [CLS] (với phoBERT thì [CLS] tương đương với </s>)
        :return: padded_input_ids (MAX_SEQ_LENGTH) , padded_token_type_ids (MAX_SEQ_LENGTH),
                    padded_lis_cls_pos (MAX_SEQ_LENGTH)
        """
        padded_lis_cls_pos = torch.full((self.max_seq_length,), -1, dtype=torch.long)

        if len(input_ids) > self.max_seq_length:
            padded_input_ids = input_ids[:self.max_seq_length]
            padded_token_type_ids = token_type_ids[:self.max_seq_length]

            filter_lis_cls_pos = torch.tensor([cls_pos for cls_pos in lis_cls_pos if cls_pos < self.max_seq_length])
            padded_lis_cls_pos[:len(filter_lis_cls_pos)] = filter_lis_cls_pos

            pad_mask = torch.ones(self.max_seq_length)
            return padded_input_ids, padded_token_type_ids, padded_lis_cls_pos, pad_mask.long()

        assert len(input_ids) == len(token_type_ids)
        num_tok = len(input_ids)

        padded_input_ids = torch.zeros(size=(self.max_seq_length,), dtype=input_ids.dtype)
        padded_input_ids[:num_tok] = input_ids

        padded_token_type_ids = torch.zeros(size=(self.max_seq_length,), dtype=token_type_ids.dtype)
        padded_token_type_ids[:num_tok] = token_type_ids

        pad_mask = torch.zeros(self.max_seq_length)
        pad_mask[:num_tok] = 1

        padded_lis_cls_pos[:len(lis_cls_pos)] = lis_cls_pos

        return padded_input_ids, padded_token_type_ids, padded_lis_cls_pos, pad_mask.long()

    def tokenizing_formatted_input(self, src, tgt=None, is_pad=False):
        """
        :param is_pad: Có padding theo MAX_SEQ_LENGTH luôn hay không
        :param src: Xâu đầu vào đã được segmentation (n_sent * n_token)
        :param tgt: Xâu đầu ra đã được segmenttion (n_sent * n_token)
        :return: src_tokenized_res (input_ids, token_type_ids, lis_cls_pos, [is_pad = true] mask)
        :return: tgt_tokenized_res (input_ids, token_type_ids, lis_cls_pos, [is_pad = true] mask)
        """
        src_tokenized_res = self.tokenize_formatted_list(src)
        tgt_tokenized_res = None
        if tgt is not None:
            tgt_tokenized_res = self.tokenize_formatted_list(tgt)

        if is_pad:
            return self.padding_seq(*src_tokenized_res), \
                   self.padding_seq(*tgt_tokenized_res) if tgt is not None else None

        return src_tokenized_res, tgt_tokenized_res

    def one_hot_lis_tgt(self, ext_id):
        one_hot_vec = torch.zeros(size=(self.max_seq_length,), dtype=torch.long)
        for i in ext_id:
            one_hot_vec[i] = 1
        return one_hot_vec

    def tokenizing_ext_input(self, src, ext_id=None, is_pad=True):
        src_tokenized_res = self.tokenize_formatted_list(src)
        one_hot_vec_tgt = None
        if ext_id is not None:
            one_hot_vec_tgt = self.one_hot_lis_tgt(ext_id)
        if not is_pad:
            return src_tokenized_res, one_hot_vec_tgt
        return self.padding_seq(*src_tokenized_res), one_hot_vec_tgt