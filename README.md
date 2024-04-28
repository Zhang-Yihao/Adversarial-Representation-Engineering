# Adversarial Representation Engineering

## Introduction
This is a minimal scale demo still in the testing phase, which provides implementation for Section 5.1 **Alignment: To Generate (Harmful Responses) or Not to Generate**. Experiment for hallucination can be simply derived by changing dataset, which will be provided after publication.

## Setup
Parameters are hardcoded in `main.py` for now. If you need to modify the parameters, please edit `main.py` directly. We will implement `argparse` in the near future.

## Execution
Currently, you can run the program by executing:

```bash
python main.py
```

You can change the model by modifying the `model_path` in `main.py`. Note that this set of parameters may not be suitable for larger models and adjustments may be necessary based on the specific circumstances.

## Dependencies
Install the necessary libraries using the following command:
```bash
pip install transformers torch numpy datasets peft pandas tqdm sklearn
```

## Additional Information
More code and details will be published following the release of our paper.


## Citation
```
@article{zhang2024towards,
  title={Towards General Conceptual Model Editing via Adversarial Representation Engineering},
  author={Zhang, Yihao and Wei, Zeming and Sun, Jun and Sun, Meng},
  journal={arXiv preprint},
  year={2024}
}
```
