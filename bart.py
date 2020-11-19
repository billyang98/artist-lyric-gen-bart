# imports
import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np

import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

import math
import random
import re
import argparse

class LitModel(pl.LightningModule):
    # Instantiate the model
    def __init__(self, learning_rate, tokenizer, model, hparams):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.learning_rate = learning_rate
        # self.freeze_encoder = freeze_encoder
        # self.freeze_embeds_ = freeze_embeds
        self.hparams = hparams

        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())

        if self.hparams.freeze_embeds:
            self.freeze_embeds()

    def freeze_embeds(self):
        ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
        freeze_params(self.model.model.shared)
        for d in [self.model.model.encoder, self.model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)

    # Do a forward pass through the model
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Load the data into variables
        src_ids, src_mask = batch[0], batch[1]
        tgt_ids = batch[2]
        # Shift the decoder tokens right (but NOT the tgt_ids)
        decoder_input_ids = shift_tokens_right(tgt_ids, tokenizer.pad_token_id)

        # Run the model and get the logits
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]
        # Create the loss function
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        # Calculate the loss on the un-shifted tokens
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        return {'loss':loss}

    def validation_step(self, batch, batch_idx):

        src_ids, src_mask = batch[0], batch[1]
        tgt_ids = batch[2]

        decoder_input_ids = shift_tokens_right(tgt_ids, tokenizer.pad_token_id)

        # Run the model and get the logits
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]

        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        val_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        return {'loss': val_loss}

    # Method that generates text using the BartForConditionalGeneration's generate() method
    def generate_text(self, text, eval_beams, early_stopping = True, max_len = 102):
        ''' Function to generate text '''
        generated_ids = self.model.generate(
            text["input_ids"],
            attention_mask=text["attention_mask"],
            use_cache=True,
            decoder_start_token_id = self.tokenizer.pad_token_id,
            num_beams= eval_beams,
            max_length = max_len,
            early_stopping = early_stopping
        )
        return [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generated_ids]

    def freeze_params(model):
        ''' Function that takes a model as input (or part of a model) and freezes the layers for faster training
            adapted from finetune.py '''
        for layer in model.parameters():
            layer.requires_grade = False

# Create a dataloading module as per the PyTorch Lightning Docs
class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_file, batch_size, num_examples = 775959):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_examples = num_examples

    # Loads and splits the data into training, validation and test sets with a 60/20/20 split
    def prepare_data(self):
        self.data = pd.read_csv(self.data_file)[:self.num_examples]
        self.train, self.validate, self.test = np.split(self.data.sample(frac=1), [621405, 621405 + 75795])

    # encode the sentences using the tokenizer
    def setup(self, stage):
        self.train = encode_sentences(self.tokenizer, self.train['source'], self.train['target'])
        self.validate = encode_sentences(self.tokenizer, self.validate['source'], self.validate['target'])
        self.test = encode_sentences(self.tokenizer, self.test['source'], self.test['target'])

    # Load the training, validation and test sets in Pytorch Dataset objects
    def train_dataloader(self):
        dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])
        train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size)
        return train_data

    def val_dataloader(self):
        dataset = TensorDataset(self.validate['input_ids'], self.validate['attention_mask'], self.validate['labels'])
        val_data = DataLoader(dataset, batch_size = self.batch_size)
        return val_data

    def test_dataloader(self):
        dataset = TensorDataset(self.test['input_ids'], self.test['attention_mask'], self.test['labels'])
        test_data = DataLoader(dataset, batch_size = self.batch_size)
        return test_data

# Create the hparams dictionary to pass in the model
# I realise that this isn't really how this is meant to be used, but having this here reminds me that I can edit it when I need
hparams = argparse.Namespace()

hparams.freeze_encoder = False
hparams.freeze_embeds = False
hparams.eval_beams = 4

def shift_tokens_right(input_ids, pad_token_id):
    """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
        This is taken directly from modeling_bart.py
    """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens

def encode_sentences(tokenizer, source_sentences, target_sentences, max_length=102, pad_to_max_length=True, return_tensors="pt"):
    ''' Function that tokenizes a sentence
        Args: tokenizer - the BART tokenizer; source and target sentences are the source and target sentences
        Returns: Dictionary with keys: input_ids, attention_mask, target_ids
    '''

    input_ids = []
    attention_masks = []
    target_ids = []
    tokenized_sentences = {}
    i = 0
    for sentence in source_sentences:
        i += 1
        sentence_tokens = sentence.split(' ')
        tok_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
        encoded_dict = tokenizer.prepare_for_model(
            tok_ids,
            max_length=max_length,
            add_special_tokens=False,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            return_attention_mask=True,
            add_prefix_space=True,
            prepend_batch_axis = True,
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)

    for sentence in target_sentences:
        sentence_tokens = sentence.split(' ')
        tok_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
        encoded_dict = tokenizer.prepare_for_model(
            tok_ids,
            max_length=max_length,
            add_special_tokens=False,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            return_attention_mask=True,
            add_prefix_space=True,
            prepend_batch_axis = True,
        )
        # Shift the target ids to the right
        # shifted_target_ids = shift_tokens_right(encoded_dict['input_ids'], tokenizer.pad_token_id)
        target_ids.append(encoded_dict['input_ids'])

    target_ids = torch.cat(target_ids, dim = 0)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": target_ids,
    }

    return batch


def noise_sentence(sentence_, percent_words, replacement_token = "<mask>"):
    '''
    Args: sentence - the sentence to noise
          percent_words - the percent of words to remove
    '''
    # Create a list item and copy
    sentence_ = sentence_.split(' ')
    newsent = []
    for word in sentence_:
        if random.random() < percent_words:
            if len(newsent) > 0:
                    if newsent[-1] != replacement_token:
                        newsent.append(replacement_token)
            else:
                newsent.append(word)
        else:
            newsent.append(word)
    return " ".join(newsent)


# Load the model
import os

from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig
myfile = open("support_files/bpe_string_token_to_int.json", "r")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", add_prefix_space=True, bos_token = "S", eos_token = "L", mask_token = "<mask>")
print(len(tokenizer))
import json
toadd = list(json.load(myfile).keys())
print(len(toadd))
# #toadd.append("<mask>")
tokenizer.add_tokens(toadd)
print(len(tokenizer))

bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
bart_model.resize_token_embeddings(len(tokenizer))

# Load the data into the model for training
# hparams for cvae:
# 256 batch size
# < 5e-3 learning rate
summary_data = SummaryDataModule(tokenizer, 'support_files/train_dataset_id.csv', batch_size = 64, num_examples = 775959)

# Load the model from a pre-saved checkpoint; alternatively use the code below to start training from scratch
# model = LitModel.load_from_checkpoint(base_dir + "checkpoint_files_2/8_ep_140k_simple_0210.ckpt",
#                                       learning_rate = 2e-5, tokenizer = tokenizer, model = bart_model, hparams = hparams)

model = LitModel(learning_rate = 2e-5, tokenizer = tokenizer, model = bart_model, hparams = hparams)

checkpoint = ModelCheckpoint(filepath='checkpoint_files/')
trainer = pl.Trainer(gpus = 1,
                     max_epochs = 1,
                     min_epochs = 1,
                     auto_lr_find = False,
                     checkpoint_callback = checkpoint,
                     progress_bar_refresh_rate = 25)


# Fit the instantiated model to the data
trainer.fit(model, summary_data)