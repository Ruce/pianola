# Pianola

[![Pianola in action](/assets/img/example-screenshot.png)](https://pianola.app)

Pianola is an application that plays AI-generated piano music. Users seed (i.e. "prompt") the AI model by playing notes on the keyboard, or choosing example snippets from classical pieces.

In this readme, we explain how the AI works and go into details about the model's architecture.

## Overview

Music can be represented in many ways, ranging from raw audio waveforms to semi-structured MIDI standards. In Pianola, we break up musical beats into regular, uniform intervals (e.g. sixteenth notes/semiquavers). Notes played within an interval are considered to belong to the same timestep, and a series of timesteps forms a sequence. Using the sequence as input, the AI model predicts the notes in the next timestep, which in turn is used as input for predicting the subsequent timestep in an autoregressive manner.

In addition to the notes to play, the model also predicts the *duration* (length of time the note is held down) and *velocity* (how hard a note is struck) of each note.

## Model Architecture

The model is made up of three modules: an embedder, an encoder, and a decoder. These modules borrow from well-known architectures such as Inception networks, Llama transformers, and multi-label classifier chains, but are adapted to work with musical data and combined in a novel approach.

### Embedder

The job of the embedder is to convert each input timestep of shape `(num_notes, num_features)` into an embedding vector that can be fed into the encoder. However, unlike text embeddings which map one-hot vectors to another dimensional space, we provide an inductive bias by applying convolutional and pooling layers on the input. We do this for a few reasons:

1. The possible combinations of notes played in a single timestep is intractably large (`2^num_notes`, where `num_notes` is 64 or 88 for short to normal pianos), therefore it is not possible to represent them as one-hot vectors.
2. We reason that the distance between notes presents additional musical information, such as chords and perfect fifths, which CNNs are able to encode.
3. CNNs are also able to synthesise information from multiple channels, which in our case are the features *duration* and *velocity*.

To allow the embedder to learn which distances are useful, we take inspiration from [Inception networks](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html) and stack convolutions of varying kernel sizes.

### Encoder

The encoder is comprised of [LLaMA transformer](https://arxiv.org/abs/2302.13971) layers that encode the sequence of input embedding vectors.

> As is common in the generative AI literature, we use the terminology "encoder" here to refer to the component that interprets the data in a high dimensional space, but the module uses a causal mask and is equivalent to a "decoder" in the original [Transformers](https://arxiv.org/abs/1706.03762) paper.

### Decoder


## Disclaimer and Notice

The source code for this project is publicly visible for the purposes of academic research and knowledge sharing. All rights are retained by the creator(s) unless permissions have been explicitly granted.

Contact us on outlook.com at the address `bruce <dot> ckc`.
