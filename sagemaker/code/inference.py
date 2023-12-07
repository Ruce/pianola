import math
import random
import re
import json
import logging
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Module

from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, AttentionMaskConverter

class InceptionConfig():
    def __init__(self, in_channels, conv1_1, conv3_1, conv5_1, conv7_1, conv1_2, conv3_2, conv5_2, conv7_2, conv1_frommax):
        self.in_channels = in_channels
        self.conv1_1 = conv1_1
        self.conv3_1 = conv3_1
        self.conv5_1 = conv5_1
        self.conv7_1 = conv7_1
        self.conv1_2 = conv1_2
        self.conv3_2 = conv3_2
        self.conv5_2 = conv5_2
        self.conv7_2 = conv7_2
        self.conv1_frommax = conv1_frommax

class InceptionModel(nn.Module):
    def __init__(self, config, num_notes, out_dim):
        assert (num_notes / 4).is_integer(), "Argument `num_notes` needs to be divisible by 4"

        super().__init__()
        self.conv1_1 = nn.Conv1d(config.in_channels, config.conv1_1, kernel_size=1, stride=1, padding=0)
        self.conv3_1 = nn.Conv1d(config.in_channels, config.conv3_1, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv1d(config.in_channels, config.conv5_1, kernel_size=5, stride=1, padding=2)
        self.conv7_1 = nn.Conv1d(config.in_channels, config.conv7_1, kernel_size=7, stride=1, padding=3)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.maxpool_1_to_2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        channel_2 = config.conv1_1 + config.conv3_1 + config.conv5_1 + config.conv7_1 + config.in_channels
        self.conv1_2 = nn.Conv1d(channel_2, config.conv1_2, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = nn.Conv1d(channel_2, config.conv3_2, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv1d(channel_2, config.conv5_2, kernel_size=5, stride=1, padding=2)
        self.conv7_2 = nn.Conv1d(channel_2, config.conv7_2, kernel_size=7, stride=1, padding=3)
        self.maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv1_frommax = nn.Conv1d(channel_2, config.conv1_frommax, kernel_size=1, stride=1, padding=0)

        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()

        hidden_dim = (config.conv1_2 + config.conv3_2 + config.conv5_2 + config.conv7_2 + config.conv1_frommax) * int(num_notes / 4)
        self.linear1 = nn.Linear(hidden_dim, out_dim)

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
        attention_mask = AttentionMaskConverter._make_causal_mask(hidden_states.shape[0:2], hidden_states.dtype, device, past_key_values_length)
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
        self.conv1 = nn.Conv1d((x_dim + chain_length) * num_notes, emb_dim * num_notes, kernel_size=1, groups=num_notes)
        self.conv2 = nn.Conv1d(emb_dim * num_notes, num_notes, kernel_size=1, groups=num_notes)
        self.update_conv_weights()

    def update_conv_weights(self):
        weights_1 = []
        biases_1 = []
        weights_2 = []
        biases_2 = []
        for classifier in self.note_classifiers:
            weights_1.append(classifier.note_linear1.weight)
            biases_1.append(classifier.note_linear1.bias)
            weights_2.append(classifier.note_linear2.weight)
            biases_2.append(classifier.note_linear2.bias)

        weights_1 = torch.cat(weights_1, dim=0).unsqueeze(dim=-1)
        biases_1 = torch.cat(biases_1, dim=0)
        weights_2 = torch.cat(weights_2, dim=0).unsqueeze(dim=-1)
        biases_2 = torch.cat(biases_2, dim=0)

        self.conv1.weight = nn.Parameter(weights_1)
        self.conv1.bias = nn.Parameter(biases_1)
        self.conv2.weight = nn.Parameter(weights_2)
        self.conv2.bias = nn.Parameter(biases_2)

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

        # Memoise the encoding of an all zeros label chain since it is most common
        allzeros_l = F.relu(self.chain_linear1(labels[..., :self.chain_length]))
        allzeros_l = self.chain_linear2(allzeros_l)
        allzeros_h = torch.cat((x, allzeros_l), dim=-1)

        # Flatten the batch and timestep dimensions into first dimension
        if len(x.shape) == 3:
            allzeros_h = allzeros_h.reshape((x.shape[0] * x.shape[1], -1))
        allzeros_h = allzeros_h.repeat((1, self.num_notes)).unsqueeze(dim=-1)
        allzeros_h = self.conv2(F.relu(self.conv1(allzeros_h))).squeeze(dim=-1)
        if len(x.shape) == 3:
            zero_prob = torch.sigmoid(allzeros_h.reshape((x.shape[0], x.shape[1], -1)))

        # Accumulate the probability masses of the note predictions:
        # If note is selected, probability = note_prob; if not selected, probability = 1-note_prob
        prob_mass = torch.zeros(x.shape[:-1]).to(device)
        for n in range(self.num_notes):
            # Encode the previous labels
            note_links = labels[..., n:n+self.chain_length]
            if (note_links == 0).all():
                note_prob = zero_prob[..., n]
            else:
                note_l = F.relu(self.chain_linear1(note_links))
                note_l = self.chain_linear2(note_l)
                h = torch.cat((x, note_l), dim=-1)
                note_prob = torch.sigmoid(self.note_classifiers[n](h).squeeze(dim=-1))
            note_pred = torch.bernoulli(note_prob)
            labels[..., n+self.chain_length] = note_pred
            prob_mass += (note_prob * note_pred) + ((1 - note_prob) * (1 - note_pred))

        # Remove padding and return predicted labels
        return labels[..., self.chain_length:], prob_mass

class FeatureModule(nn.Module):
    def __init__(self, num_notes, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, num_notes)

    def forward(self, x):
        h = F.relu(self.linear_1(x))
        h = self.linear_2(h)
        return h

class LalaE(nn.Module):
    def __init__(self, config, inception_config, num_notes, chain_emb_dim, chain_length, dropout=0.1):
        super().__init__()
        self.config = config
        self.incep = InceptionModel(inception_config, num_notes, config.hidden_size)
        self.lalama = Lalama(config, dropout)
        self.chain = ChainClassifier(num_notes, config.hidden_size, chain_emb_dim, chain_length)
        self.velocity = FeatureModule(num_notes, config.hidden_size)
        self.duration = FeatureModule(num_notes, config.hidden_size)

    def train_model(self, source, labels):
        src = self.incep(source)
        lalama_out = self.lalama(src)
        note_pred = self.chain.train_model(lalama_out[0], labels)
        velocity_pred = self.velocity(lalama_out[0])
        duration_pred = self.duration(lalama_out[0])
        return (note_pred, velocity_pred, duration_pred)

    def forward(self, source, past_key_values=None, last_step_only=False):
        src = self.incep(source)
        lalama_out = self.lalama(src, past_key_values)
        representation = lalama_out[0]
        if last_step_only:
            representation = representation[:, -1:]
        note_pred, prob_mass = self.chain(representation)
        velocity_pred = torch.clamp(self.velocity(representation), min=0, max=1)
        duration_pred = torch.clamp(self.duration(representation), min=0)
        output = (note_pred, prob_mass, velocity_pred, duration_pred, lalama_out[1]) if self.config.use_cache and not self.training else (note_pred, prob_mass, velocity_pred, duration_pred)
        return output

    def get_past_key_values(self, source, past_key_values=None):
        src = self.incep(source)
        lalama_out = self.lalama(src, past_key_values)
        return lalama_out[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_NOTES = 64
WINDOW_SIZE = 384
TIMESTEPS_PER_CHUNK = 16
OPTIONS_DELIMITER = '_'
BASE_52_MAPPING = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def to_base_52(number):
    def decimal_to_custom_base(number, target_base, str_mapping):
        custom_number = ""
        while number > 0:
            remainder = number % target_base
            custom_number = str_mapping[remainder] + custom_number
            number = number // target_base
        return custom_number
    return decimal_to_custom_base(number, 52, BASE_52_MAPPING)
  
def from_base_52(number_str):
    def custom_base_to_decimal(number_str, source_base, str_mapping):
        total = 0
        for i, c in enumerate(reversed(number_str)):
            value = str_mapping.index(c)
            total += value * int(math.pow(source_base, i))
        return total
    return custom_base_to_decimal(number_str, 52, BASE_52_MAPPING)

def feature_tensor_to_str(feature_tensors):
    '''
        Converts a tensor of shape (num_timesteps, num_notes, 2) to a string. Last dimension of tensor contains features (velocity, duration)
        Output is a delimited string, where each segment is separated by a number, and each segment is a timestep containing active notes in that step
        Active notes are represented as 4 character long base52 strings, which when converted to base10 are 7 digit long numbers
        Of the base10 number, the first two digits is the note number (shifted by 2), next two digits is velocity (shifted by 1), and last three digits is duration (scaled by 10)
        E.g. 'KjJs' = 5088062: note = 50 - 2 = 48, velocity = 0.89, duration = 6.2
    '''
    outputs = []
    for feature_tensor in feature_tensors:
        notes_list = []
        for t in range(len(feature_tensor)):
            slice_note_str = ''
            tensor_slice = feature_tensor[t]
            active_notes_in_slice = torch.nonzero(torch.sum(tensor_slice, dim=-1), as_tuple=True)[0].tolist()
            for n in active_notes_in_slice:
                # Transform velocity to a 2 digit integer ranging from 0 to 99
                velocity = round((tensor_slice[n, 0].item() * 100) - 1)
                velocity = max(min(velocity, 99), 0)

                # Scale duration up by 10 so it is a 3 digit integer ranging from 0 to 999
                duration = round(tensor_slice[n, 1].item() * 10)
                duration = max(min(duration, 999), 1)

                # N.B. Important: we shift the note up by 2 so that the base52 string is guaranteed to be 4 characters long
                note_shifted = n + 2

                number_str = f"{note_shifted:02d}{velocity:02d}{duration:03d}"
                slice_note_str += to_base_52(int(number_str))
            notes_list.append(slice_note_str)
        output = ''
        spaces = 0
        for notes_str in notes_list:
            if notes_str == '':
                spaces += 1
            else:
                # Add the number of spaces before this timestep of active notes
                output += str(spaces) + notes_str
                spaces = 0
        if spaces > 0:
            output += str(spaces)
        outputs.append(output)
    return OPTIONS_DELIMITER.join(outputs)
  
def notes_str_to_feature_tensor(notes_str, num_notes):
    notes_slices = [x for x in re.split("\d+", notes_str) if x != '']
    spaces = [int(x) for x in re.findall("\d+", notes_str)]
    notes_tensor = torch.zeros((sum(spaces) + len(notes_slices), num_notes, 2), dtype=torch.float)

    t = 0
    for active_notes_str, space in zip(notes_slices, spaces):
        t += space
        active_notes = [active_notes_str[i:i+4] for i in range(0, len(active_notes_str), 4)]
        for note_str in active_notes:
            note_numerals = f"{from_base_52(note_str):07d}"
            note_num = int(note_numerals[0:2]) - 2 # N.B. Important: the note was shifted up by 2 when converting to a note string
            velocity = (int(note_numerals[2:4]) + 1) / 100
            duration = int(note_numerals[4:7]) / 10
            notes_tensor[t, note_num, 0] = velocity
            notes_tensor[t, note_num, 1] = duration
        t += 1
    return notes_tensor

def select_pkv(pkv, idx):
    selected_pkv = []
    for layer in pkv:
        selected_pkv.append([cache[idx:idx+1] for cache in layer])
    return selected_pkv

def repeat_pkv(pkv, num_repeats):
    repeat_shape = [num_repeats] + [1] * (len(pkv[0][0].shape) - 1) # Subtract 1 here because first dimension is the batch, which we will repeat
    repeated_pkv = []
    for layer in pkv:
        new_layer = [cache.repeat(repeat_shape) for cache in layer]
        repeated_pkv.append(new_layer)
    return repeated_pkv

def precompute_pkv(model, x, num_repeats):
    '''
        Input `x` shape: (1, timesteps, num_notes, 2), where the first dimension is to be repeated
        Returns `past_key_values` of type Tuple(Tuple(Tensor[num_repeats, num_attn_heads, seq_len, head_dim]))
    '''
    with torch.no_grad():
        pkv = model.get_past_key_values(x)
        return repeat_pkv(pkv, num_repeats)

def generate_music(model, seed, timesteps, num_repeats=1, selection_idx=0, past_key_values=None):
    '''
        Input `seed` and output shapes: (timesteps, num_notes, 2), where the last dimension has features (velocity, duration)
    '''
    assert num_repeats >= 1, 'Argument `num_repeats` cannot be less than 1'
    
    model.eval()
    if num_repeats == 1:
        source = seed.unsqueeze(dim=0)
    else:
        if past_key_values is None:
            # Prior to repeating the seed, cache the results of all timesteps minus the last one
            seed_to_precompute = seed[:-1].unsqueeze(dim=0)
            past_key_values = precompute_pkv(model, seed_to_precompute, num_repeats)
        else:
            past_key_values = repeat_pkv(past_key_values, num_repeats)
        source = seed[-1:].unsqueeze(dim=0).repeat(num_repeats, 1, 1, 1)
    
    generated = []
    prob_masses = []

    with torch.no_grad():
        for i in range(timesteps):
            note_pred, prob_mass, velocity_pred, duration_pred, past_key_values = model(source, past_key_values, last_step_only=True)
            # Use note_pred as a mask to select feature values
            velocity = note_pred * velocity_pred
            duration = note_pred * duration_pred
            source = torch.stack((velocity, duration), dim=-1)
            generated.append(source)
            prob_masses.append(prob_mass)
    generated = torch.cat(generated, dim=1)
    prob_masses = torch.cat(prob_masses, dim=1).sum(dim=1)
    if selection_idx < 0:
        # Return all generated samples with their corresponding prob and cache values
        return generated, prob_masses, past_key_values
    else:
        sample_idx = prob_masses.argsort()[selection_idx]
        return generated[sample_idx]

def generate_options(model, seed, timesteps, timesteps_per_chunk, base_num_repeats):
    # Split the timesteps into chunks so that we get more branching options
    base_timesteps = min(timesteps_per_chunk, timesteps)

    # Halve the number of repeats when generating options to speed up process
    options_num_repeats = max(base_num_repeats // 2, 1)
    options_timesteps = max(timesteps - base_timesteps, 0)
    
    # Generate base samples to branch from
    generated, prob_masses, past_key_values = generate_music(model, seed, base_timesteps, base_num_repeats, selection_idx=-1)
    
    ## Get 3 options:
    # 1. Lowest duration
    # 2. Lowest probability mass (most chaotic)
    # 3. Highest probability mass (most stable)

    durations = torch.tensor([(sample[..., 1] > 0).sum() for sample in generated])
    duration_idx = durations.argsort()[0]

    prob_mass_indices = prob_masses.argsort().tolist()
    prob_mass_indices.remove(duration_idx)
    low_prob_idx = prob_mass_indices[0] if len(prob_mass_indices) > 0 else duration_idx
    high_prob_idx = prob_mass_indices[-1] if len(prob_mass_indices) > 0 else duration_idx

    duration_gen = generated[duration_idx]
    low_prob_gen = generated[low_prob_idx]
    high_prob_gen = generated[high_prob_idx]

    def generate_branched_option(option_generated, option_selection, pkv):
        timesteps = options_timesteps
        while timesteps > 0:
            curr_timesteps = min(timesteps_per_chunk, timesteps)
            timesteps -= curr_timesteps

            new_gen, new_prob_masses, new_pkv = generate_music(model, option_generated[-1:], curr_timesteps, options_num_repeats, selection_idx=-1, past_key_values=pkv)
            new_durations = torch.tensor([(sample[..., 1] > 0).sum() for sample in new_gen])
            match option_selection:
                case 'duration':
                    option_idx = new_durations.argsort()[0]
                case 'low_prob':
                    option_idx = new_prob_masses.argsort()[0]
                case 'high_prob':
                    option_idx = new_prob_masses.argsort()[-1]
                case _:
                    raise ValueError(f"Unknown option {option_selection}")
            
            option_generated = torch.cat((option_generated, new_gen[option_idx]))
            pkv = select_pkv(new_pkv, option_idx)
        return option_generated

    duration_gen = generate_branched_option(duration_gen, 'duration', select_pkv(past_key_values, duration_idx))
    low_prob_gen = generate_branched_option(low_prob_gen, 'low_prob', select_pkv(past_key_values, low_prob_idx))
    high_prob_gen = generate_branched_option(high_prob_gen, 'high_prob', select_pkv(past_key_values, high_prob_idx))
    return duration_gen, low_prob_gen, high_prob_gen

# Define model and load weights
def model_fn(model_dir):
    config = LlamaConfig(
        vocab_size=0,
        hidden_size=384,
        intermediate_size=2048,
        num_hidden_layers=16,
        num_attention_heads=12,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_theta=10000.0,
        rope_scaling=None,
    )

    inception_config = InceptionConfig(
        in_channels = 2, conv1_1 = 4, conv3_1 = 12, conv5_1 = 10, conv7_1 = 8,
        conv1_2 = 12, conv3_2 = 24, conv5_2 = 16, conv7_2 = 12, conv1_frommax = 8
    )
    
    model = LalaE(config, inception_config, NUM_NOTES, chain_emb_dim=32, chain_length=12, dropout=0.0)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device))
    model.to(device).eval()
    return model


# Preprocess input data
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    request_json = json.loads(request_body)
    notes_str = request_json["inputs"]
    timesteps = request_json["timesteps"]
    num_repeats = request_json["num_repeats"]
    selection_idx = request_json["selection_idx"]
    data = notes_str_to_feature_tensor(notes_str, num_notes=NUM_NOTES)
    return {'data': data, 'timesteps': timesteps, 'num_repeats': num_repeats, 'selection_idx': selection_idx}


# Retrieve prediction parameters from input
def predict_fn(input_object, model):
    timesteps = int(input_object['timesteps'])
    num_repeats = int(input_object['num_repeats'])
    selection_idx = int(input_object['selection_idx'])
    # Trim input_object['data'] tensor to a maximum length of WINDOW_SIZE
    seed = input_object['data'][-WINDOW_SIZE:]
    if selection_idx >= 0:
        prediction = generate_music(model, seed, timesteps, num_repeats, selection_idx)
        predictions = (prediction,)
    else:
        predictions = generate_options(model, seed, timesteps, TIMESTEPS_PER_CHUNK, num_repeats)
    return predictions


# Return generated output as json
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    res = feature_tensor_to_str(predictions)
    return json.dumps(res)