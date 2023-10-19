# Pianola

![Pianola in action](/assets/img/example-screenshot.png)

Pianola is an application that plays AI-generated piano music. Users seed (i.e. "prompt") the AI model by playing notes on the keyboard, or choosing example snippets from classical pieces.

In this readme, we explain how the AI works and describe the design choices behind the model's architecture.

## Overview

Music can be represented in many ways, ranging from raw audio waveforms to semi-structured MIDI standards.

For Pianola, we use a simple representation that breaks up musical beats into regular, uniform intervals (e.g. sixteenth notes/semiquavers). Notes played within an interval are considered to belong to the same timestep, and a series of timesteps forms a sequence. Using the sequence as input, the AI model predicts the notes in the next timestep, which is then used as input for predicting the subsequent timestep in an autoregressive manner.

In addition to predicting the notes to play, the model also determines the *duration* (length of time the note is held down) and *velocity* (how hard a note is struck) of each note.

## Model Architecture

The model is made up of three modules: an embedder, an encoder, and a decoder. These modules borrow from well-known architectures such as Inception networks, Llama transformers, and multi-label classifier chains, which we explain in turn.

### Embedder

The job of the embedder is to convert each input timestep of shape `(num_notes, num_features)` into an embedding vector that can be fed into the encoder. However, unlike text embeddings which map one-hot vectors to another dimensional space, we provide an inductive bias by applying convolutional and pooling layers on the input. We do this for two reasons:

1. The possible combinations of notes played in a single timestep is intractably large (`2^num_notes`, where `num_notes` is 64 or 88 for short to normal pianos), therefore it is not possible to represent them as one-hot vectors.
2. We reason that the distance between notes presents additional musical information, such as chords and perfect fifths, which CNNs are able to encode.

To allow the embedder to learn which distances are useful, we take inspiration from Inception networks and stack convolutions of varying kernel sizes.

### Encoder


### Decoder


## Disclaimer and Notice

The source code for this project is publicly visible for the purposes of academic research and knowledge sharing. All rights are retained by the creator(s) unless permissions have been explicitly granted.
