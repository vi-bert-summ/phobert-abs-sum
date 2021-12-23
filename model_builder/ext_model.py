import torch

from pytorch_lightning import LightningModule
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss

from model_builder.ext_bert_sum import ExtBertSumm


class ExtBertSummPylight(LightningModule):

    def __init__(self, learning_rate: float = 2e-3
                 , adam_epsilon: float = 1e-8
                 , warmup_steps: int = 1000
                 , weight_decay: float = 0.01
                 , **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.total_steps = 10
        self.model = ExtBertSumm()
        self.loss_fn = CrossEntropyLoss()

    def forward(self, src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask):
        out_logits = self.model(src_ids=src_inp_ids, src_pad_mask=src_mask,
                                src_token_type=src_tok_type_ids,
                                src_cls_pos=src_lis_cls_pos)
        return out_logits

    def training_step(self, batch, batch_idx):
        """
        :param batch: input dataloader (src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask
                        , tgt_inp_ids, tgt_tok_type_ids, tgt_lis_cls_pos, tgt_mask)
        :param batch_idx: Số thứ tự của batch
        :return:
        """
        src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask, ext_ids = batch
        out_logits = self.model(src_ids=src_inp_ids, src_pad_mask=src_mask
                                , src_token_type=src_tok_type_ids
                                , src_cls_pos=src_lis_cls_pos)
        out_prob = torch.sigmoid(out_logits).reshape(len(src_inp_ids), -1)
        masked_out_prob = out_prob * src_mask

        loss = self.loss_fn(masked_out_prob, ext_ids.float())

        tensorboard_logs = {'train_loss': loss.detach()}
        return {'loss': loss, 'log': tensorboard_logs}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask, ext_ids = batch
        with torch.no_grad():
            out_logits = self.model(src_ids=src_inp_ids, src_pad_mask=src_mask
                                    , src_token_type=src_tok_type_ids
                                    , src_cls_pos=src_lis_cls_pos)
            out_prob = torch.sigmoid(out_logits).reshape(len(src_inp_ids), -1)
            masked_out_prob = out_prob * src_mask
            return masked_out_prob

    def configure_optimizers(self):
        """
        Chuẩn bị Optimizer với chiến lược warm-up và weight-decay
        """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-3, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=1000,
            num_training_steps=10
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask, ext_ids = batch
        with torch.no_grad():
            out_logits = self.model(src_ids=src_inp_ids, src_pad_mask=src_mask
                                    , src_token_type=src_tok_type_ids
                                    , src_cls_pos=src_lis_cls_pos)
            out_prob = torch.sigmoid(out_logits).reshape(len(src_inp_ids), -1)
            masked_out_prob = out_prob * src_mask

            loss = self.loss_fn(masked_out_prob, ext_ids.float())

            tensorboard_logs = {'val_loss': loss}
            self.log('val_loss', loss.detach())
            return {'loss': loss, 'log': tensorboard_logs}