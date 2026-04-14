# Model Card: miklium-lm-nano

## Model Details
- **Developer**: OpenAGI for the MIKLIUM ecosystem
- **Model Type**: Decoder-only Transformer Language Model
- **Architecture**: Custom, built entirely in pure Python / NumPy
- **Total Parameters**: 502.7K
- **Number of Layers**: 4
- **Embedding Dimension**: 64
- **Attention Heads**: 4
- **Context Length (Block Size)**: 128 tokens
- **License**: MIT License

## Intended Use
`miklium-lm-nano` is the most compact and accessible model within the MIKLIUM LM family. It is intended as a foundation model designed for rapid deployment, educational purposes, and experimentation. The model is capable of demonstrating foundational reasoning and chain-of-thought (`<think>`) capabilities.

## Architecture & Infrastructure
- **Framework**: Custom NumPy-based neural network backbone, ensuring maximum portability and transparency without external deep learning libraries like PyTorch or TensorFlow.
- **Normalization**: Root Mean Square Normalization (RMSNorm)
- **Weight Initialization**: Xavier Initialization
- **Activation Function**: ReLU
- **Tokenization**: Custom word-level tokenization supporting specific model tags (`<user>`, `<ai>`, `<think>`, `</think>`, `<eos>`).

## Training Pipeline
- **Dataset**: Curated dataset, trained using autoregressive next-token prediction.
- **Optimizer**: Adam (beta1=0.9, beta2=0.95, eps=1e-8)
- **Learning Rate**: 0.008 with a linear decay schedule over 5000 steps
- **Batch Size**: 4

## Limitations
Being an extremely small "nano" model (~500K parameters), `miklium-lm-nano` is fundamentally a proof-of-concept and educational foundation model. It will require additional fine-tuning, expanded datasets, and further training epochs to unlock higher-tier linguistic and reasoning capabilities.
