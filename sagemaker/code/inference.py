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

class ConvModel(nn.Module):
    def __init__(self, num_notes, first_channel, second_channel, emb_dim, out_dim):
        super().__init__()
        self.conv3_1 = nn.Conv1d(1, first_channel, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv1d(first_channel, second_channel, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv1d(1, first_channel, kernel_size=5, stride=1, padding=2)
        self.conv5_2 = nn.Conv1d(first_channel, second_channel, kernel_size=5, stride=1, padding=2)
        self.conv7_1 = nn.Conv1d(1, first_channel, kernel_size=7, stride=1, padding=3)
        self.conv7_2 = nn.Conv1d(first_channel, second_channel, kernel_size=7, stride=1, padding=3)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        
        linear1_in_dim = int(num_notes * 3 * second_channel / 4)
        self.linear1 = nn.Linear(linear1_in_dim, emb_dim)
        self.linear2 = nn.Linear(emb_dim, out_dim)

    def forward(self, x):
        h = torch.reshape(x, (x.shape[0] * x.shape[1], 1, -1))

        h3 = F.relu(self.conv3_1(h))
        h3 = self.pool(h3)
        h3 = F.relu(self.conv3_2(h3))
        h3 = self.pool(h3)
        h3 = self.flatten(h3)

        h5 = F.relu(self.conv5_1(h))
        h5 = self.pool(h5)
        h5 = F.relu(self.conv5_2(h5))
        h5 = self.pool(h5)
        h5 = self.flatten(h5)

        h7 = F.relu(self.conv7_1(h))
        h7 = self.pool(h7)
        h7 = F.relu(self.conv7_2(h7))
        h7 = self.pool(h7)
        h7 = self.flatten(h7)

        pos = torch.cat((h3, h5, h7), dim=-1)
        pos = F.relu(self.linear1(pos))
        pos = self.linear2(pos)
        pos = torch.reshape(pos, (x.shape[0], x.shape[1], -1))
        return pos

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
    
class ConvLalama(nn.Module):
    def __init__(self, config, num_notes, first_channel, second_channel, conv_emb_dim, dropout=0.1):
        super().__init__()
        self.config = config
        self.conv_model = ConvModel(num_notes, first_channel, second_channel, conv_emb_dim, out_dim=config.hidden_size)
        self.lalama = Lalama(config, dropout)
        self.linear = nn.Linear(config.hidden_size, num_notes)

    def forward(self, source, past_key_values=None):
        src = self.conv_model(source)
        lalama_out = self.lalama(src, past_key_values)
        pred = self.linear(lalama_out[0])
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
  
def decode_tensor(y_hat, max_notes):
    assert len(y_hat.shape) == 1
    sample = torch.bernoulli(y_hat)
    if torch.count_nonzero(sample) > max_notes:
        sample_prob = y_hat * sample
        to_keep = torch.argsort(sample_prob, descending=True)[:max_notes]
        sample = torch.zeros(y_hat.shape)
        sample[to_keep] = 1
    return sample

def generate_music(model, seed, timesteps, max_notes=6):
    # Input `seed` and output shapes: (timesteps, num_notes)
    # For convolutional models, expected input shape is (batch_size, timesteps, num_notes, 1)
    source = seed.unsqueeze(dim=0).unsqueeze(dim=-1)
    generated = []

    model.eval()
    with torch.no_grad():
        past_key_values = None
        for i in range(timesteps):
            pred, past_key_values = model(source, past_key_values)
            y_hat = torch.sigmoid(pred[:, -1]).squeeze(dim=0) # Keep only the last timestep and remove batch_size dimension
            new_notes = decode_tensor(y_hat, max_notes) # Decode probabilities in y_hat
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

    model = ConvLalama(config, NUM_NOTES, first_channel=8, second_channel=32, conv_emb_dim=512)
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