import logging
import mido
import math
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MidiDataset(Dataset):
  def __init__(self, tensors, source_size, sample_delta=1, note_shift=0):
    '''
    `tensors`: list of tensors where each tensor is a song of shape (timesteps, notes)
    `source_size`: number of timesteps for the source sequence
    `sample_delta`: number of timesteps between each sample, i.e. overlapping samples if < source_size, gaps between samples if > source_size
    `note_shift`: randomly shift the notes up or down by this amount in order to augment data. Results in some data loss as tensors will padded and trimmed
    '''
    self.tensors = tensors
    self.tensor_lengths = [len(t) for t in tensors]
    self.source_size = source_size
    self.sample_delta = sample_delta
    self.note_shifts = list(range(-note_shift, note_shift+1))
    # Calculate number of samples per tensor
    self.tensor_samples = [max(0, ((n - self.source_size - 1) // self.sample_delta) + 1) for n in self.tensor_lengths]

  def __len__(self):
    return sum(self.tensor_samples)

  def __getitem__(self, idx):
    curr_sample_id = 0
    random.seed(idx) # Set the seed according to idx for replicability
    note_shift = random.choice(self.note_shifts)

    for i, n in enumerate(self.tensor_samples):
      curr_sample_id += n
      if curr_sample_id > idx:
        # This tensor contains the item we want
        tensor = self.tensors[i]

        # Which window in this sample contains the item we want?
        window_idx = idx - (curr_sample_id - n)
        start = window_idx * self.sample_delta
        end = start + self.source_size

        if note_shift > 0:
          tensor = F.pad(tensor, (note_shift, 0))
          tensor = tensor[:, :-note_shift]
        elif note_shift < 0:
          tensor = F.pad(tensor, (0, abs(note_shift)))
          tensor = tensor[:, abs(note_shift):]
        return tensor[start:end].unsqueeze(dim=-1), tensor[start+1:end+1]

class MidiUtil():
  def get_midi_timesteps(filename):
    midi = mido.MidiFile(filename)
    assert midi.type == 0 or midi.type == 1, "Type 2 MIDI files are not supported"
    timesteps = 0
    for track in midi.tracks:
      track_timesteps = 0
      for msg in track:
        track_timesteps += msg.time
      timesteps = max(timesteps, track_timesteps)
    return timesteps

  def midi_to_opo_tensor(filename):
    midi = mido.MidiFile(filename)

    # Three types of MIDI files (0, 1, 2)
    # This function works with types 0 and 1 only, and smooshes all tracks into one
    # https://mido.readthedocs.io/en/stable/files/midi.html#file-types
    assert midi.type == 0 or midi.type == 1, "Type 2 MIDI files are not supported"
    
    max_timesteps = MidiUtil.get_midi_timesteps(filename)
    tensor = np.zeros((max_timesteps+1, 128)) # 128 notes as in MIDI specifications
    for track in midi.tracks:
      timesteps = 0
      for msg in track:
        timesteps += msg.time
        if msg.type == 'note_on':
          tensor[timesteps, msg.note] = msg.velocity
    return torch.from_numpy(tensor)

  def to_binary_velocity_tensor(tensor):
    return (tensor > 0).long()

  def calculate_compress_factor(orig_tpb, desired_tpb):
    compress_factor = orig_tpb / desired_tpb
    if compress_factor % 1.0 != 0.0:
      logging.warning(f"compress_factor of {compress_factor} is not an integer, rounding up...")
    compress_factor = math.ceil(compress_factor)
    return compress_factor

  def compress_tensor(tensor, method, orig_tpb, desired_tpb=16):
    '''
    Reduces the fidelity of the musical tensor, i.e. merge multiple timesteps into one step

    Args:
      `tensor`: PyTorch tensor of shape (timesteps, num_notes)
      `method`: str in ["max", "avg", "majority"]
      `orig_tpb`: original ticks per beat of the input `tensor`
      `desired_tpb`: desired ticks per beat for the tensor to be compressed to
    '''
    assert(len(tensor.shape) == 2)

    compress_factor = MidiUtil.calculate_compress_factor(orig_tpb, desired_tpb)
    compressed_vectors = []
    length = tensor.shape[0]
    for start in range(0, length, compress_factor):
      end = min(start + compress_factor, length)
      tensor_slice = tensor[start:end, :]
      if (method == "max"):
        compressed_vectors.append(tensor_slice.max(dim=0).values)
      elif (method == "avg"):
        raise NotImplementedError()
      elif (method == "majority"):
        majority = (end-start) / 2
        majority_nonzeroes = np.count_nonzero(tensor_slice, axis=0) >= majority
        compressed_vectors.append(torch.tensor((majority_nonzeroes).astype(int)))
      else:
        raise KeyError(f"Unknown method {method}")
    return torch.stack(compressed_vectors)

  def reduce_tensor(tensor, start_note, end_note):
    '''
    Args:
      `tensor`: PyTorch tensor of shape (timesteps, num_notes)
      `start_note`: note to start the tensor from (integer in 0-127)
      `end_note`: note to end the tensor at (integer in 0-127)
    '''
    assert(end_note >= start_note)
    return tensor[:, start_note:end_note+1]

  def uncompress_tensor(tensor, orig_tpb, compressed_tpb):
    '''
    Args:
      `tensor`: PyTorch tensor of shape (timesteps, num_notes)
      `orig_tpb`: ticks per beat of the original/generated MIDI file
      `compressed_tpb`: ticks per beat used by the compressed `tensor`
    '''
    compress_factor = orig_tpb / compressed_tpb
    if compress_factor % 1.0 != 0.0:
      logging.warning(f"compress_factor of {compress_factor} is not an integer, rounding up...")
    compress_factor = math.ceil(compress_factor)
    # "Stretch" out the tensor using Kronecker product
    return torch.kron(tensor, torch.ones((compress_factor, 1)))

  def unreduce_tensor(tensor, start_note, end_note):
    '''
    Expands out a reduced tensor to include all 128 notes in the MIDI range

    Args:
      `tensor`: PyTorch tensor of shape (timesteps, num_notes)
      `start_note`: MIDI note that `tensor` starts from (integer in 0-127)
      `end_note`: MIDI note that `tensor` ends at (integer in 0-127)
    '''
    assert(end_note >= start_note)
    timesteps = tensor.shape[0]
    low_notes = torch.zeros((timesteps, start_note)).to(DEVICE)
    high_notes = torch.zeros((timesteps, 127-end_note)).to(DEVICE)
    return torch.cat((low_notes, tensor, high_notes), dim=1)

  def slice_temporal_data(timestep_tensor, window_size):
    '''
    Slice music sequence based on window_size (history length) to return training data
    Input shape (num_timesteps, num_notes)
    Output shapes (num_notes, num_timesteps)
    '''
    num_slices = len(timestep_tensor) - window_size
    notes_tensor = timestep_tensor.transpose(0, 1)
    return [[notes_tensor[:, i:i+window_size], notes_tensor[:, i+window_size]] for i in range(num_slices)]


class TonnetzUtil():
  # In Tonnetz, each node has six neighbours which have pitches of the following distances (in semi-tones)
  # E.g. C4 has neighbours F3, G#3, A3, D#4, E4, G4
  NEIGHBOUR_DISTANCES = [-7, -4, -3, 3, 4, 7]

  def create_tonnetz_adjacency_matrix(num_notes):
    # In Tonnetz, each node has six neighbours which have pitches of the following distances (in semi-tones)
    # E.g. C4 has neighbours F3, G#3, A3, D#4, E4, G4
    A = []
    for i in range(num_notes):
      row = torch.zeros(num_notes, dtype=torch.int)
      for d in TonnetzUtil.NEIGHBOUR_DISTANCES:
        j = i+d
        if j >= 0 and j < num_notes:
            row[j] = 1
      A.append(row)
    A = torch.stack(A)
    # Check that A is symmetric since the Tonnetz graph is undirected
    assert(torch.equal(A, A.transpose(0, 1)))

    # Convert to sparse format expected by PyG layers
    edge_index = A.to_sparse().indices()
    return edge_index

  def create_tonnetz_edge_attr(edge_index):
    edge_attr_indices = []
    for i in range(edge_index.shape[1]):
        distance = (edge_index[1][i] - edge_index[0][i]).item()
        edge_attr_indices.append(TonnetzUtil.NEIGHBOUR_DISTANCES.index(distance))

    edge_attr = F.one_hot(torch.tensor(edge_attr_indices))
    return edge_attr