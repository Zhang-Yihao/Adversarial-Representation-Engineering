import torch
from torch.cuda.amp import autocast
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    @autocast()
    def forward(self, x):
        x = self.fc(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    @autocast()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        #x = nn.functional.softmax(x, dim=1)
        return x


class CombinedModel(nn.Module):
    def __init__(self, generator_model, classifier_model, layer_range):
        super(CombinedModel, self).__init__()
        self.generator = generator_model
        self.classifier = classifier_model
        self.layer_range = layer_range

    @autocast()
    def forward(self, prompts):
        # 使用Llama2生成文本
        generated_texts = self.generator(prompts, output_hidden_states=1)
        now_hs = generated_texts.hidden_states[self.layer_range[0]][:, -1, :]
        for i in self.layer_range[1:]:
            now_hs = torch.concat((now_hs, generated_texts.hidden_states[i][:, -1, :]), dim=0)
        inputs = now_hs.flatten().unsqueeze(0)
        classification_outputs = self.classifier(inputs)
        return classification_outputs


def get_simple_classifier(input_size, num_classes, half=True):
    model = SimpleClassifier(input_size, num_classes)
    if half:
        model.half()
    return model


def get_linear_classifier(input_size, num_classes, half=True):
    model = LinearClassifier(input_size, num_classes)
    if half:
        model.half()
    return model


def get_combined_model(generator_model, classifier_model, layer_range):
    return CombinedModel(generator_model, classifier_model, layer_range)
