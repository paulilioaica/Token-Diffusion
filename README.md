# 🚀 Conditional Diffusion Sequence Model

## Overview

This project implements a **Conditional Diffusion Sequence Model** for text generation. Inspired by diffusion models used in image generation, this model starts with a noisy sequence and iteratively refines it to generate coherent text. The process is guided by input tokens, which condition the denoising process and help steer the model towards generating meaningful sequences.

This is still work in progress.


### ✨ Key Features

- 🔄 **Diffusion-Inspired Process**: The model begins with random noise and gradually reduces the noise over multiple iterations, refining the sequence at each step.
- 🔍 **Transformer Decoder**: A Transformer decoder is used to iteratively denoise the sequence, conditioned on the input tokens.
- 🛠️ **Flexible Sequence Generation**: The model can be used for various text generation tasks, such as text completion, sequence inpainting, and more.

## 📦 Installation

To run the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/paulilioaica/Token-Diffusion
```

## 📊 Diffusion Process Diagram

The diagram below illustrates the diffusion process used in this model. The process starts with a noisy sequence and iteratively refines it over several steps, guided by the input tokens.

```
 +----------+      +----------+      +----------+      +----------+
 |  Noisy   |      | Less     |      | Further  |      | Final    |
 | Sequence | ---> | Noisy    | ---> | Refined  | ---> | Sequence |
 |  (Noise) |      | Sequence |      | Sequence |      |          |
 +----------+      +----------+      +----------+      +----------+
      |                |                |                |
      v                v                v                v
   Noise         Transformer        Transformer       Transformer
 Reduction         Decoder            Decoder           Decoder
```

## 🤝 Contributing

Feel free to submit issues or pull requests. Contributions are welcome!

## 📜 License

This project is licensed under the MIT License