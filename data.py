from datasets import load_dataset
import torch
import numpy as np
import pandas as pd
import random
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


def load_function_dataset(func_name):
    func_name_dict = {
        "jailbreak": jailbreak_dataset,
    }
    return func_name_dict[func_name]

# Other datasets is not well supported, so here we only provide jailbreak dataset

def jailbreak_dataset(data_path, template):
    dataset = load_dataset(data_path)

    train_dataset, test_dataset = dataset['train'], dataset['test'] if 'test' in dataset else dataset['train']

    train_data, train_labels = train_dataset['sentence'], train_dataset['label']
    test_data = test_dataset['sentence']

    train_data = np.concatenate(train_data).tolist()
    test_data = np.concatenate(test_data).tolist()

    train_data = [template.format(instruction=s) for s in train_data]
    test_data = [template.format(instruction=s) for s in test_data]
    test_labels = test_dataset['label']
    train_labels = torch.tensor(train_labels).flatten().long()
    test_labels = torch.tensor(test_labels).flatten().long()
    return train_data, train_labels, test_data[:100], test_labels[:100]

def get_ft_dataloader(train_data, test_data, tokenizer, label=0):
    train_loader_data = []
    test_loader_data = []
    for s in train_data:
        train_loader_data.append(tokenizer.encode(s, return_tensors="pt"))
    for s in test_data:
        test_loader_data.append(tokenizer.encode(s, return_tensors="pt"))
    if label == 0:
        train_zero_labels = torch.zeros([len(train_data), 1], dtype=torch.long)
        test_zero_labels = torch.zeros([len(test_data), 1], dtype=torch.long)
    else:
        train_zero_labels = torch.ones([len(train_data), 1], dtype=torch.long)
        test_zero_labels = torch.ones([len(test_data), 1], dtype=torch.long)

    train_loader = list(zip(train_loader_data, train_zero_labels))
    test_loader = list(zip(test_loader_data, test_zero_labels))
    return train_loader, test_loader