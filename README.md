# `promoterai-pytorch`

PyTorch implementation of [PromoterAI](https://github.com/Illumina/PromoterAI), a deep learning model developed by Illumina to predict the effect of promoter mutations on gene expression.

**Note**: This is a preliminary implementation under development.

## Overview

PromoterAI models promoter regions using a stack of convolutional blocks (MetaFormer) and shallow fully connected heads. It supports inference in a **twin network configuration** to quantify differential effects of sequence changes.

This reimplementation follows Hugging Face `transformers` conventions with a configuration class (`PromoterAIConfig`), making integration and hyperparameter management more convenient.

## Usage

```python
import torch
from promoterai_pytorch import PromoterAI, TwinWrapper

# Define configuration using PromoterAI training defaults
config = PromoterAIConfig()

# Instantiate the base model
model = PromoterAI(config)

# Wrap in twin network for delta predictions
twin = TwinWrapper(model)

# Create dummy reference and alternate sequences
ref = torch.randn(8, config.input_length, 4)  # one-hot encoded 4-channel input
alt = torch.randn(8, config.input_length, 4)

# Forward pass through the twin model
output = twin(ref, alt)  # Shape: [8]
print("Twin delta output shape:", output.shape)
````

## Installation

```shell
pip install git+https://github.com/MiqG/promoterai-pytorch.git
```

## References
- Jaganathan, K., Ersaro, N., Novakovsky, G., Wang, Y., James, T., Schwartzentruber, J., Fiziev, P., Kassam, I., Cao, F., Hawe, J. and Cavanagh, H., 2025. Predicting expression-altering promoter mutations with deep learning. Science, p.eads7373. URL: https://www.science.org/doi/abs/10.1126/science.ads7373

- Original repo (Tensorflow): https://github.com/Illumina/PromoterAI/tree/master

## TODO
- [ ] double check architecture matching
- [ ] training and finetuning scripts
