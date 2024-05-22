from transformers import AutoTokenizer, AutoConfig, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import torch.nn.functional as F
import gc


def batchify(lst, batch_size):
    """Yield successive batch_size chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def load_tqa_sentences(user_tag, assistant_tag, preset=""):
    dataset = load_dataset('./tqa/multiple_choice')['validation']
    questions, answers = [],[]
    labels = []
    for d in dataset:
        q = d['question']
        for i in range(len(d['mc1_targets']['labels'])):
            a = d['mc1_targets']['choices'][i]
            questions = [f'{user_tag}' + q + ' ' + preset] + questions
            answers = [f'{assistant_tag}' + a] + answers
        ls = d['mc1_targets']['labels']
        ls.reverse()
        labels.insert(0, ls)
    return questions, answers, labels

def get_logprobs(logits, input_ids, masks, **kwargs):
    logprobs = F.log_softmax(logits, dim=-1)[:, :-1]
    # find the logprob of the input ids that actually come next in the sentence
    logprobs = torch.gather(logprobs, -1, input_ids[:, 1:, None])
    logprobs = logprobs * masks[:, 1:, None] 
    return logprobs.squeeze(-1)
    
def prepare_decoder_only_inputs(prompts, targets, tokenizer, device):
    tokenizer.padding_side = "left"
    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    tokenizer.padding_side = "right"
    target_inputs = tokenizer(targets, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
    
    # concatenate prompt and target tokens and send to device
    inputs = {k: torch.cat([prompt_inputs[k], target_inputs[k]], dim=1).to(device) for k in prompt_inputs}

    # mask is zero for padding tokens
    mask = inputs["attention_mask"].clone()
    # set mask to 0 for question tokens
    mask[:, :prompt_inputs["input_ids"].shape[1]] = 0
    mask.to(device)
    # remove token_type_ids
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    return inputs, mask, prompt_inputs["input_ids"].shape[1]

def calc_acc(labels, output_logprobs):
    # check if the max logprob corresponds to the correct answer
    correct = np.zeros(len(labels))
    # indices to index
    indices = np.cumsum([len(l) for l in labels])
    indices = np.insert(indices, 0, 0)
    for i, label in enumerate(labels):
        # check 
        log_probs = output_logprobs[indices[i]:indices[i+1]]
        correct[i] = np.argmax(log_probs) == label.index(1)
    return correct.mean()

def get_tqa_accuracy(model, questions, answers, labels, tokenizer, batch_size=128):
    gc.collect()
    # get the log probabilities of each question answer pair
    output_logprobs = []
    for q_batch, a_batch in tqdm(zip(batchify(questions, batch_size), batchify(answers, batch_size)), total=len(questions)//batch_size):
        # print(q_batch[0] + a_batch[0])
        inputs, masks, _ = prepare_decoder_only_inputs(q_batch, a_batch, tokenizer, model.model.device)

        with torch.no_grad():
            try:
                # set the masks so that we do not add to tokens of input sentences and padding tokens
                model.set_masks(masks.unsqueeze(-1))
            except:
                pass

            # calculate the probabilities for all tokens (all question answer pairs)
            logits = model(**inputs).logits
            # sum the probabilities for each question answer pair so that each pair has one probability
            # mask is zero for question and padding tokens
            logprobs = get_logprobs(logits, inputs['input_ids'], masks).sum(-1).detach().cpu().numpy()
        output_logprobs.extend(logprobs)

    return calc_acc(labels, output_logprobs)