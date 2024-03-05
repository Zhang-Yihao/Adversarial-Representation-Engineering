from datasets import load_dataset
import torch
import numpy as np
import pandas as pd
import random
from transformers import AutoTokenizer


def load_function_dataset(func_name):
    func_name_dict = {
        "jailbreak": jailbreak_dataset,
        "jailbreak_answer": jailbreak_answer_dataset,
        "honesty": honesty_function_dataset,
        "emotion": emotion_function_dataset
    }
    return func_name_dict[func_name]


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


def jailbreak_answer_dataset(data_path,
                             user_tag="[INST]",
                             assistant_tag="[/INST]",
                             template=None):
    if template is None:
        template = user_tag + "{prompt}" + assistant_tag + "{answer}"
    dataset = load_dataset(data_path)
    jailbreak = dataset['train']['chosen']
    no_jailbreak = dataset['train']['rejected']
    prompt = dataset['train']['prompt']
    total_dataset = []
    total_labels = []

    for i in range(len(prompt)):
        total_dataset.append(template.format(prompt=prompt[i], answer=jailbreak[i]))
        total_labels.append(0)
        total_dataset.append(template.format(prompt=prompt[i], answer=no_jailbreak[i]))
        total_labels.append(1)

    train_data = total_dataset[:100]
    train_labels = torch.tensor(total_labels[:100])
    test_data = total_dataset[100:150]
    test_labels = torch.tensor(total_labels[100:150])

    return train_data, train_labels, test_data, test_labels


def honesty_function_dataset(data_path: str, tokenizer: AutoTokenizer, user_tag: str = "", assistant_tag: str = "",
                             seed: int = 0) -> (list, list, list, list):
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Train Data, Train Labels, Test Data, Test Labels
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df['label'] == 1]['statement'].values.tolist()
    false_statements = df[df['label'] == 0]['statement'].values.tolist()

    template_str = "Pretend you're {type} person making statements about the world."
    honest_statements = []
    untruthful_statements = []

    # Process statements
    for statement in true_statements:
        tokens = tokenizer.tokenize(statement)

        for idx in range(1, len(tokens) - 5):
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)

            honest_statements.append(
                f"{user_tag} {template_str.format(type='an honest')} {assistant_tag} " + truncated_statement)
            untruthful_statements.append(
                f"{user_tag} {template_str.format(type='an untruthful')} {assistant_tag} " + truncated_statement)

    # Create training data
    ntrain = 512
    combined_data = [[honest, untruthful] for honest, untruthful in zip(honest_statements, untruthful_statements)]
    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])

    train_data = np.concatenate(train_data).tolist()

    # Create test data
    reshaped_data = np.array([[honest, untruthful] for honest, untruthful in
                              zip(honest_statements[:-1], untruthful_statements[1:])]).flatten()
    test_data = reshaped_data[ntrain:ntrain * 2].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    dataset = {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1, 0]] * (len(test_data) // 2)}
    }

    train_data, test_data = dataset['train']['data'], dataset['test']['data']
    train_labels = torch.tensor(dataset['train']['labels']).flatten().long()
    test_labels = torch.tensor(dataset['test']['labels']).flatten().long()

    return train_data, train_labels, test_data, test_labels


def emotion_function_dataset(data_path: str, tokenizer: AutoTokenizer, user_tag: str = "", assistant_tag: str = "",
                             seed: int = 0,
                             target_emotion="angry", orig_emotion="happy") -> (list, list):
    """
    Processes data to create training and testing datasets based on emotion.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.
    - target_emotion (str): Target emotion.
    - orig_emotion (str): Original emotion.

    Returns:
    - Train Data, Train Labels, Test Data, Test Labels
    """
    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    emo_orig_statements = df[df['label'] == 1]['statement'].values.tolist()
    emo_target_statements = df[df['label'] == 0]['statement'].values.tolist()

    template_str = " Act as if you are extremely {emo}."
    orig_statements = []
    target_statements = []

    # Create training data
    # Process statements
    for statement in emo_orig_statements:
        orig_statements.append(f"{user_tag} {template_str.format(emo='happy')} {assistant_tag} " + statement)

    for statement in emo_target_statements:
        target_statements.append(f"{user_tag} {template_str.format(emo='angry')} {assistant_tag} " + statement)

    # Create training data
    ntrain = 300
    combined_data = [[orig, target] for orig, target in zip(orig_statements, target_statements)]
    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])

    train_data = np.concatenate(train_data).tolist()

    # Create test data
    reshaped_data = np.array([[honest, untruthful] for honest, untruthful in
                              zip(orig_statements[:-1], target_statements[1:])]).flatten()
    test_data = reshaped_data[ntrain:ntrain * 2].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    dataset = {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1, 0]] * (len(test_data) // 2)}
    }

    train_data, test_data = dataset['train']['data'], dataset['test']['data']
    train_labels = torch.tensor(dataset['train']['labels']).flatten().long()
    test_labels = torch.tensor(dataset['test']['labels']).flatten().long()
    return train_data, train_labels, test_data, test_labels


def get_ft_dataloader(train_data, test_data, tokenizer):
    train_loader_data = []
    test_loader_data = []
    for s in train_data:
        train_loader_data.append(tokenizer.encode(s, return_tensors="pt"))
    for s in test_data:
        test_loader_data.append(tokenizer.encode(s, return_tensors="pt"))

    train_zero_labels = torch.ones([len(train_data),1],dtype=torch.long)
    test_zero_labels = torch.ones([len(test_data),1],dtype=torch.long)

    train_loader = list(zip(train_loader_data, train_zero_labels))
    test_loader = list(zip(test_loader_data, test_zero_labels))
    return train_loader, test_loader

