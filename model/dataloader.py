import os
import copy
from typing import Optional, Dict, Sequence
import torch
from datasets import load_dataset
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence

def load_dataset_from_path(save_data_dir, dataset_name, train_file, validation_file, test_file):
    save_data_dir = os.path.join(save_data_dir, dataset_name.split("/")[-1])

    print(f"Load data from: {save_data_dir}")
    train_file_path = os.path.join(save_data_dir, train_file)
    train_dataset = load_dataset('json', data_files=train_file_path)

    validation_file_path = os.path.join(save_data_dir, validation_file)
    validation_dataset = load_dataset('json', data_files=validation_file_path)

    test_file_path = os.path.join(save_data_dir, test_file)
    test_dataset = load_dataset('json', data_files=test_file_path)

    return {
        'train': train_dataset['train'],
        'validation': validation_dataset['train'],
        'test': test_dataset['train']
    }

def preprocess_function(examples, data_args, tokenizer):
    # Tokenize the texts
    source_max_len = min(data_args.source_max_len, tokenizer.model_max_length)
    model_inputs = tokenizer(
        examples["sentence"], 
        padding="max_length", 
        max_length=source_max_len, 
        truncation=True
    )
    labels = tokenizer(
        examples["label"],
        padding=True
    )
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    return model_inputs