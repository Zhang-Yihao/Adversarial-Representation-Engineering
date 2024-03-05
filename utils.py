import torch
from tqdm import tqdm


def get_logits_from_sentence(model, tokenizer, sentence):
    with torch.no_grad():
        tokenized_input = tokenizer.encode(sentence, return_tensors="pt").to(model.device)
        logits = model(tokenized_input).logits
        return logits


def get_response_from_sentence(model, tokenizer, sentence, max_length=250):
    with torch.no_grad():
        tokenized_input = tokenizer.encode(sentence, return_tensors="pt").to(model.device)
        generated = model.generate(tokenized_input, max_length=max_length, do_sample=0)
        return tokenizer.decode(generated[0][len(tokenized_input):])


def hidden_state_generator(model, tokenizer, data, layer_range=range(18, 32)):
    hidden_states = []
    with torch.no_grad():
        for dt in tqdm(data):
            tokenized_input = tokenizer(dt, return_tensors="pt").input_ids.to(model.device)
            hidden_st = model(tokenized_input, output_hidden_states=1).hidden_states
            now_hs = hidden_st[layer_range[0]][:, -1, :]
            for i in layer_range[1:]:
                now_hs = torch.concat((now_hs, hidden_st[i][:, -1, :]), dim=0)
            hidden_states.append(now_hs.flatten())
    hidden_states = torch.stack(hidden_states)
    return hidden_states.squeeze().to(model.device)
