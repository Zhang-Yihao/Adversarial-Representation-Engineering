# Adversarial Representation Engineering

## Introduction
This is a minimal scale demo still in the testing phase.

## Setup
`argparse` is not implemented for now, and parameters are hardcoded in `main.py`. If you need to modify the parameters, please edit `main.py` directly.

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