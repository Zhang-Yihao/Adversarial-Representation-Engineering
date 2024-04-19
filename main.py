# Description: This is the main script for the PEFT demo.
# For testing issue, one good way is to run it in jupyter notebook or colab.

from transformers import LlamaTokenizer, AutoModelForCausalLM
import torch
from torch import fx
from torch.utils.data import DataLoader, TensorDataset
from utils import get_response_from_sentence, hidden_state_generator
from data import load_function_dataset, get_ft_dataloader
from model import get_linear_classifier, get_simple_classifier, get_combined_model
from train import classifier_trainer, peft_model_finetune
import peft
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

# The model and tokenizer path is defined here.
# Please change the path to your own model path. Parameters are set for Llama-2 7B model, so it should be adjusted for other models.
model_path = "./meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Change torch.float16 for float32 fine-tuning, while discriminator should be halfed
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16)
device = "cuda:0"

# For Llama-2 7B. For best performance this should be adjusted to similar ranges for bigger model
layer_range = range(-18, -23, -1)

# In this demo, other funcs are not supported
func_name = "jailbreak"
data_path = "justinphan3110/harmful_harmless_instructions"

# 0 = harmless, 1 = harmful
target_label = 1
dataloader_func = load_function_dataset(func_name)
discriminator_choice = {"linear": get_linear_classifier, "simple": get_simple_classifier}

# empirical value. In the first time, one can try higher epochs and determine the epoch num by observing the output
epoch_num = 50
discriminator_type = "simple"

# Learning rate for the lora adapter. Should be better designed.
learning_rate_schedule = [1e-4] * epoch_num
# target bound is layers that is to be tuned
# for example, (2m, 2n) = q_proj and v_proj in layers (m,n)
target_bound = (18, 28)

# generator training epochs for each peft iteration
generator_epoch = 2
# discriminator training epochs for each peft iteration
discriminator_epoch = 30

get_discriminator = discriminator_choice[discriminator_type]

user_tag = "[INST]"
assistant_tag = "[/INST]"

template = user_tag + " {instruction} " + assistant_tag

config_log = f"""
Model Path: {model_path}
Layer Range: {layer_range}
Function Name: {func_name}
Data Path: {data_path}
Epoch Number: {epoch_num}
Learning Rate Schedule: {learning_rate_schedule}
Target Bound: {target_bound}
User_tag: {user_tag}
Assistant_tag: {assistant_tag}
Template: {template}
Discriminator_type: {discriminator_type}
Discriminator_epoch: {discriminator_epoch}
Target_label: {target_label}
"""
print(config_log)

train_data, train_labels, test_data, test_labels = dataloader_func(data_path, template=template)  # tokenizer,user_tag,assistant_tag)#, template=template)
print("Generating training data...")
train_hs = hidden_state_generator(model, tokenizer, train_data, layer_range)
print("Training data generated. Shape:", train_hs.shape)

print("Generating testing data...")
test_hs = hidden_state_generator(model, tokenizer, test_data, layer_range)
print("Testing data generated. Shape:", test_hs.shape)

train_labels = train_labels.cuda()
test_labels = test_labels.cuda()
train_dataset = TensorDataset(train_hs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
input_size = train_hs.shape[1]

classifier = get_discriminator(input_size, 2, half=True)
classifier = classifier.to(model.device)
classifier_trainer(classifier, train_loader, epochs=discriminator_epoch, device=model.device,
                   is_eval=True, eval_dset=test_hs, eval_labels=test_labels)

combined_model = get_combined_model(model, classifier, layer_range)

modules = [name for name, _ in combined_model.named_modules()][1:]
target_modules = [name for name in modules if "q_proj" in name or "v_proj" in name][target_bound[0]:target_bound[1]]
print("Target modules:", target_modules)
lora_config = LoraConfig(target_modules=target_modules)
peft_model = get_peft_model(combined_model, lora_config)
peft_model.print_trainable_parameters()

train_loader, test_loader = get_ft_dataloader(train_data, test_data, tokenizer, target_label)

for i in range(epoch_num):
    peft_model_finetune(peft_model, train_loader, test_loader, epochs=2, lr=learning_rate_schedule[i], eps=1e-2,
                        device=device)
    print("Generating training data " + str(i))
    ctrain_hs = hidden_state_generator(peft_model.generator, tokenizer, train_data, layer_range)
    print("Training data generated. Shape:", train_hs.shape)

    print("Generating testing data " + str(i))
    ctest_hs = hidden_state_generator(peft_model.generator, tokenizer, test_data, layer_range)
    print("Testing data generated. Shape:", test_hs.shape)

    ctrain_dataset = TensorDataset(ctrain_hs, train_labels)
    ctrain_loader = DataLoader(ctrain_dataset, batch_size=64, shuffle=True)

    classifier = get_discriminator(input_size, 2, half=True)
    classifier = classifier.to(model.device)
    converge = classifier_trainer(classifier, ctrain_loader, epochs=discriminator_epoch, device=model.device,
                                  is_eval=True, eval_dset=ctest_hs, eval_labels=test_labels)
    if converge:
        break
    peft_model.classifier.load_state_dict(classifier.state_dict())

response_list = []
for prompt in tqdm(test_data, desc="Evaluating Editing Results"):
    response = get_response_from_sentence(model, tokenizer, prompt, max_length=256)
    print(response)