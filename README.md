# Adversarial Representation Engineering (NeurIPS 2024)

This is the official implementation repository for the paper **Adversarial Representation Engineering: A General Model Editing Framework for Large Language Models** (NeurIPS 2024), [arxiv](https://arxiv.org/abs/2404.13752)). See details below.

w/ [Yihao Zhang](https://zhang-yihao.github.io), [Zeming Wei](https://weizeming.github.io), Jun Sun, Meng Sun.

## Introduction
This minimal scale demo is still in the testing phase, which provides the implementation for Section 5.1 **Alignment: To Generate (Harmful Responses) or Not to Generate** and 5.2 **Hallucination: To Hallucinate or Not to Hallucinate**.

## Setup
Parameters are hardcoded in `main.py` for now. If you wish to modify the parameters, please edit `main.py` directly. We will implement `argparse` soon.

## Execution
Currently, you can run the program by executing:

```bash
python main.py
```

You can change the model by modifying the `model_path` in `main.py`. Please note that this set of parameters may not be suitable for larger models, and adjustments may be necessary based on the specific requirements.
Demo for decreasing hallucination is provided in `hallucination.ipynb`.

## Dependencies
Install the necessary libraries including:
```bash
transformers
torch>=2.0
numpy
datasets
peft
pandas
tqdm
sklearn
```

## Additional Information
More code and details will be available upon publication of our paper.
Code for processing *TrustfulQA* dataset is partly borrowed from [This Repo](https://github.com/andyzoujm/representation-engineering).

## Citation
```
@InProceedings{zhang2024towards,
  title={Adversarial Representation Engineering: A General Model Editing Framework for Large Language Models},
  author={Zhang, Yihao and Wei, Zeming and Sun, Jun and Sun, Meng},
  booktitle = {NeurIPS},
  year={2024}
}
```
