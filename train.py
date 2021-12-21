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

@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "whether to use adafactor"})
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear", metadata={"help": f"Which lr scheduler to use."}
    )

def process_data_to_model_inputs(batch):      
    """
    Input: 
        - tokenizer: AutoTokenizer - pre train
        - batch: batch data
        - encoder_max_length: max lenth of input to summary
        - decoder_max_length: max lenth of output 
    Output: batch data preprocess
    """                                                         
    # Tokenizer will automatically set [BOS] <text> [EOS]                                               
    inputs = tokenizer(batch["original"], padding="max_length", truncation=True, max_length=args.eml)
    outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=args.dml)
                                                                                                        
    batch["input_ids"] = inputs.input_ids                                                               
    batch["attention_mask"] = inputs.attention_mask                                                     
    batch["decoder_input_ids"] = outputs.input_ids                                                      
    batch["labels"] = outputs.input_ids.copy()                                                          
    # mask loss for padding                                                                             
    batch["labels"] = [                                                                                 
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    ]                     
    batch["decoder_attention_mask"] = outputs.attention_mask                                                                              
                                                                                                         
    return batch  

# load rouge for validation
rouge = datasets.load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

if __name__ == '__main__':
    rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g') 

    # get path file data
    train_paths = utils.listPaths('vietnews-master/data/train_tokenized/*')
    val_paths = utils.listPaths('vietnews-master/data/val_tokenized/*')
    test_paths = utils.listPaths('vietnews-master/data/test_tokenized/*')

    # convert to DataFrame
    train_df = utils.get_dataframe(train_paths)
    val_df = utils.get_dataframe(val_paths)
    test_df = utils.get_dataframe(test_paths)

    # tokenizer
    # phobert = AutoModel.from_pretrained("vinai/phobert-base")
    # For transformers v4.x+: 
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

    # Dataset form fr fine-tuning huggingface
    train_data =  Dataset.from_pandas(train_df)
    val_data =  Dataset.from_pandas(val_df)
    test_data =  Dataset.from_pandas(test_df)

    # Dataset batch
    train_data_batch = train_data.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=16, 
        remove_columns=["file", "original", "summary"],
    )
    train_data_batch.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )

    val_data_batch = val_data.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=16, 
        remove_columns=["file", "original", "summary"],
    )
    val_data_batch.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    ) 

    # train
    # set encoder decoder tying to True
    roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained("vinai/phobert-base", "vinai/phobert-base", tie_encoder_decoder=True)

    # set special tokens
    roberta_shared.config.decoder_start_token_id = tokenizer.bos_token_id                                             
    roberta_shared.config.eos_token_id = tokenizer.eos_token_id
    # roberta_shared.config.pad_token_id = tokenizer.eos_token_id

    # sensible parameters for beam search
    # set decoding params                               
    roberta_shared.config.max_length = args.dml
    roberta_shared.config.early_stopping = True
    roberta_shared.config.no_repeat_ngram_size = 3
    roberta_shared.config.length_penalty = 2.0
    roberta_shared.config.num_beams = 4
    roberta_shared.config.vocab_size = roberta_shared.config.encoder.vocab_size  

    # set training arguments - these params are not really tuned, feel free to change
    training_args = Seq2SeqTrainingArguments(
        output_dir='./',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        # evaluate_during_training=True,
        do_train=True,
        do_eval=True,
        logging_steps=200,  # 2000 for full training
        save_steps=1500,  #  500 for full training
        eval_steps=7500,  # 7500 for full training
        warmup_steps=3000,  # 3000 for full training
        num_train_epochs=args.epochs, #no comment for full training
        overwrite_output_dir=True,
        save_total_limit=50,
        # fp16=True, 
    )

    # roberta_shared.config.vocab_size = roberta_shared.config.encoder.vocab_size

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=roberta_shared,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data_batch,
        eval_dataset=val_data_batch,
    )

    trainer.train()
    if not os.path.isdir(args.save_model):
        os.mkdir(args.save_model)
    trainer.save_model(args.save_model)