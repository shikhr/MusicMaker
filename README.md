# Music Generation with Transformer Model

This project aims to generate new musical compositions in the MIDI format using a transformer model trained on a dataset of piano MIDI files. The generated MIDI files can be converted to audio files, allowing you to listen to the model's compositions.

## Overview

The project consists of the following main components:

1. **Tokenizer Training**: A tokenizer is trained on the MIDI dataset to convert the MIDI files into a sequence of tokens that can be fed into the transformer model.

2. **Model Training**: A transformer model is trained on the tokenized MIDI data to learn the patterns and structures present in the dataset.

3. **Music Generation**: The trained model is used to generate new MIDI sequences, which can be converted to audio files for listening.

## Features

- **Tokenization**: The project uses the `miditok` library to tokenize MIDI files, allowing the model to learn from the musical data more effectively.
- **Transformer Architecture**: The model architecture is based on the transformer decoder, which has shown excellent performance in various sequence-to-sequence tasks, including language modeling and music generation.
- **MIDI to Audio Conversion**: The generated MIDI files can be converted to audio files (e.g., WAV) using the `midi2audio` library, enabling you to listen to the model's compositions.

## Usage

1. **Install Dependencies**: Install the required Python packages by running `pip install -r requirements.txt`.

2. **Train the Tokenizer**: Run `train_tokenizer.py` to train the tokenizer on the MIDI dataset. This step will create tokenizers of different vocabulary sizes and save them to the Hugging Face Hub. You can change the code to save them locally as well.

3. **Train the Model**: Run `train_model.py` to train the transformer model on the tokenized MIDI dataset. The trained model weights will be saved as `m1.pt`.

4. **Generate Music**: Run `generate.py` to generate new MIDI files using the trained model. This script will load the tokenizer and the trained model weights, generate a new MIDI sequence, and convert it to an audio file (`generated.wav`).

## Dataset

The dataset used for training the model is the `adl-piano-midi` dataset, which is a collection of piano MIDI files from [this paper](https://arxiv.org/abs/2008.07009)

## References

```
@article{ferreira_aiide_2020,
title={Computer-Generated Music for Tabletop Role-Playing Games},
author={Ferreira, Lucas N and Lelis, Levi HS and Whitehead, Jim},
booktitle = {Proceedings of the 16th AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment},
series = {AIIDE'20},
year={2020},
}
```
