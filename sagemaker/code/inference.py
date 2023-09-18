import json
import logging
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Module, Sequential

from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, _make_causal_mask

class InceptionModel(nn.Module):
    def __init__(self, num_notes):
        super().__init__()
        self.conv1_1 = nn.Conv1d(1, 3, kernel_size=1, stride=1, padding=0)
        self.conv3_1 = nn.Conv1d(1, 10, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv1d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv7_1 = nn.Conv1d(1, 4, kernel_size=7, stride=1, padding=3)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.maxpool_1_to_2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv1_2 = nn.Conv1d(24, 8, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = nn.Conv1d(24, 18, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv1d(24, 12, kernel_size=5, stride=1, padding=2)
        self.conv7_2 = nn.Conv1d(24, 6, kernel_size=7, stride=1, padding=3)
        self.maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv1_frommax = nn.Conv1d(24, 4, kernel_size=1, stride=1, padding=0)

        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(768, 128)

    def forward(self, x):
        # Flatten the batch and timestep dimensions into first dimension, and swap the features and channels dimensions
        h = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[-1], -1))

        h1_1 = F.relu(self.conv1_1(h))
        h3_1 = F.relu(self.conv3_1(h))
        h5_1 = F.relu(self.conv5_1(h))
        h7_1 = F.relu(self.conv7_1(h))
        max_1 = self.maxpool_1(h)

        h_1 = torch.cat((h1_1, h3_1, h5_1, h7_1, max_1), dim=-2)
        h_1 = self.maxpool_1_to_2(h_1)

        h1_2 = F.relu(self.conv1_2(h_1))
        h3_2 = F.relu(self.conv3_2(h_1))
        h5_2 = F.relu(self.conv5_2(h_1))
        h7_2 = F.relu(self.conv7_2(h_1))
        max_2 = self.maxpool_2(h_1)
        max_2 = F.relu(self.conv1_frommax(max_2))

        h_2 = torch.cat((h1_2, h3_2, h5_2, h7_2, max_2), dim=-2)
        h_2 = self.avgpool(h_2)

        h_3 = self.flatten(h_2)
        h_3 = self.linear1(h_3)
        h_3 = torch.reshape(h_3, (x.shape[0], x.shape[1], -1))
        return h_3

class Lalama(nn.Module):
    def __init__(self, config, dropout):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_key_values=None):
        '''
            `past_key_values`: a list of `num_hidden_layers` length,where each element is a tuple containing the cached `key` and `value` tensors of shape (batch_size, num_heads, seq_len, hidden_dim)
        '''
        hidden_states = x
        seq_len = hidden_states.shape[-2]

        # Sequence length of the cached keys and values
        past_key_values_length = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
        attention_mask = _make_causal_mask(hidden_states.shape[0:2], hidden_states.dtype, device, past_key_values_length)
        position_id = torch.arange(start=past_key_values_length, end=past_key_values_length+seq_len, dtype=torch.long, device=device)
        position_ids = position_id.repeat(hidden_states.shape[0], 1) # Repeat position_id for each sample in the batch

        present_key_values = []
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_key_value = layer(hidden_states, attention_mask, position_ids, past_key_value, use_cache=self.config.use_cache)
            hidden_states = self.dropout(hidden_states)
            present_key_values.append(present_key_value)

        # Only return cache key_values if not training, because they are not consistent from dropout during training
        output = (hidden_states, present_key_values) if self.config.use_cache and not self.training else (hidden_states, )
        return output

class NoteClassifier(nn.Module):
    def __init__(self, x_dim, emb_dim, chain_length):
        super().__init__()
        self.note_linear1 = nn.Linear(x_dim + chain_length, emb_dim)
        self.note_linear2 = nn.Linear(emb_dim, 1)

    def forward(self, x):
        h = F.relu(self.note_linear1(x))
        h = self.note_linear2(h)
        return h

class ChainClassifier(nn.Module):
    def __init__(self, num_notes, x_dim, emb_dim, chain_length):
        super().__init__()
        self.num_notes = num_notes
        self.chain_length = chain_length
        self.chain_linear1 = nn.Linear(chain_length, chain_length)
        self.chain_linear2 = nn.Linear(chain_length, chain_length)
        self.note_classifiers = torch.nn.ModuleList()
        for _ in range(num_notes):
            self.note_classifiers.append(NoteClassifier(x_dim, emb_dim, chain_length))

    def train_model(self, x, labels):
        assert labels.shape[-1] == self.num_notes

        # Get the previous `self.chain_length` labels for each note n
        labels = F.pad(labels, (self.chain_length, 0))
        links = torch.stack([labels[..., n:n+self.chain_length] for n in range(self.num_notes)], dim=-2)

        # Encode the previous labels
        l = F.relu(self.chain_linear1(links))
        l = self.chain_linear2(l)

        out = []
        for n in range(self.num_notes):
            note_l = l[..., n, :]
            h = torch.cat((x, note_l), dim=-1)
            out.append(self.note_classifiers[n](h))
        return torch.cat(out, dim=-1)

    def forward(self, x):
        # To get labels shape, replace last dim of x (feature space) with `num_notes` and padding `chain_length`
        labels_shape = list(x.shape)
        labels_shape[-1] = self.num_notes + self.chain_length
        labels = torch.zeros(labels_shape).to(device)

        for n in range(self.num_notes):
            # Encode the previous labels
            note_links = labels[..., n:n+self.chain_length]
            note_l = F.relu(self.chain_linear1(note_links))
            note_l = self.chain_linear2(note_l)

            h = torch.cat((x, note_l), dim=-1)
            note_pred = self.note_classifiers[n](h).squeeze(dim=-1)
            labels[..., n+self.chain_length] = torch.bernoulli(torch.sigmoid(note_pred))

        # Remove padding and return predicted labels
        return labels[..., self.chain_length:]

class LalaC(nn.Module):
    def __init__(self, config, num_notes, chain_emb_dim, chain_length, dropout=0.1):
        super().__init__()
        self.config = config
        self.incep = InceptionModel(num_notes)
        self.lalama = Lalama(config, dropout)
        self.chain = ChainClassifier(num_notes, config.hidden_size, chain_emb_dim, chain_length)

    def train_model(self, source, labels):
        src = self.incep(source)
        lalama_out = self.lalama(src)
        pred = self.chain.train_model(lalama_out[0], labels)
        return (pred, )

    def forward(self, source, past_key_values=None):
        src = self.incep(source)
        lalama_out = self.lalama(src, past_key_values)
        pred = self.chain(lalama_out[0])
        output = (pred, lalama_out[1]) if self.config.use_cache and not self.training else (pred, )
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_NOTES = 64
WINDOW_SIZE = 128
NOTES_DELIMITER = ','
TICKS_DELIMITER = ';'

def notes_tensor_to_str(notes_tensor):
    ticks = notes_tensor.shape[0]

    notes_list = []
    for t in range(ticks):
        active_notes_in_slice = torch.nonzero(notes_tensor[t], as_tuple=True)[0]
        notes_list.append(NOTES_DELIMITER.join([str(n) for n in active_notes_in_slice.tolist()])) # Convert tensor to list to string
    return TICKS_DELIMITER.join(notes_list)

def notes_str_to_tensor(notes_str, num_notes):
    notes_slices = notes_str.split(TICKS_DELIMITER)
    notes_tensor = torch.zeros((len(notes_slices), num_notes))

    for t, active_notes_str in enumerate(notes_slices):
        if len(active_notes_str) != 0:
            active_notes = [int(n) for n in active_notes_str.split(NOTES_DELIMITER)]
            notes_tensor[t, active_notes] = 1
    return notes_tensor

def generate_music(model, seed, timesteps):
    # Input `seed` and output shapes: (timesteps, num_notes)
    # For convolutional models, expected input shape is (batch_size, timesteps, num_notes, 1)
    source = seed.unsqueeze(dim=0).unsqueeze(dim=-1)
    generated = []

    model.eval()
    with torch.no_grad():
        past_key_values = None
        for i in range(timesteps):
            pred, past_key_values = model(source, past_key_values)
            new_notes = pred[:, -1].squeeze(dim=0) # Keep only the last timestep and remove batch_size dimension
            generated.append(new_notes)
            source = new_notes.reshape((1, 1, -1, 1))
    return torch.stack(generated)

# defining model and loading weights to it.
def model_fn(model_dir):
    config = LlamaConfig(
        vocab_size=0,
        hidden_size=128,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_theta=10000.0,
        rope_scaling=None,
    )

    model = LalaC(config, NUM_NOTES, chain_emb_dim=32, chain_length=12, dropout=0.0)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device))
    model.to(device).eval()
    return model


# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    request_json = json.loads(request_body)
    notes_str = request_json["inputs"]
    timesteps = request_json["timesteps"]
    data = notes_str_to_tensor(notes_str, num_notes=NUM_NOTES)
    return {'data': data, 'timesteps': timesteps}


# inference
def predict_fn(input_object, model):
    timesteps = int(input_object['timesteps'])
    # Trim input_object['data'] tensor to a maximum length of WINDOW_SIZE
    seed = input_object['data'][-WINDOW_SIZE:]
    prediction = generate_music(model, seed, timesteps)
    return prediction


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    res = notes_tensor_to_str(predictions)
    return json.dumps(res)