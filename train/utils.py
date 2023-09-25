import logging
import mido
import math
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import circmean, circvar

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
          padding = (note_shift, 0) if len(tensor.shape) == 2 else (0, 0, note_shift, 0)
          tensor = F.pad(tensor, padding)
          tensor = tensor[:, :-note_shift]
        elif note_shift < 0:
          padding = (0, abs(note_shift)) if len(tensor.shape) == 2 else (0, 0, 0, abs(note_shift))
          tensor = F.pad(tensor, padding)
          tensor = tensor[:, abs(note_shift):]
        x = tensor[start:end]
        if len(x.shape) == 2:
          x = x.unsqueeze(dim=-1) # Add a channel dimension
        y = tensor[start+1:end+1]
        return x, y

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
    
  def midi_to_timed_tensor(filename):
    '''
      Returns a tensor of shape [timesteps, num_notes, 2], where the last dimension is (velocity, duration)
    '''
    def deactivate_notes(notes_last_active, notes_to_deactivate, t):
      for idx in notes_to_deactivate:
        last_active = notes_last_active[idx]
        note_duration = t - last_active
        output[last_active, idx, 1] = note_duration
        notes_last_active[idx] = -1
    
    midi = mido.MidiFile(filename)

    # Three types of MIDI files (0, 1, 2)
    # This function works with types 0 and 1 only, and smooshes all tracks into one
    # https://mido.readthedocs.io/en/stable/files/midi.html#file-types
    assert midi.type == 0 or midi.type == 1, "Type 2 MIDI files are not supported"
    
    max_timesteps = MidiUtil.get_midi_timesteps(filename)
    # 128 notes as in MIDI specifications; last dimension is (velocity, duration)
    output = np.zeros((max_timesteps, 128, 2))
    for track in midi.tracks:
      t = 0
      sustain_active = False
      notes_last_active = np.full((128), -1, dtype=int)
      notes_is_held = np.full((128), False, dtype=bool)
      for msg in track:
        t += msg.time
        if msg.type == 'control_change' and msg.control == 64:
          if msg.value >= 64 and not sustain_active:
            # Sustain pedal activated
            sustain_active = True
          elif msg.value < 64 and sustain_active:
            # Sustain pedal deactivated; for notes that are not held down but are active, deactivate them
            sustain_active = False
            notes_to_deactivate = (np.invert(notes_is_held) * (notes_last_active != -1)).nonzero()[0]
            deactivate_notes(notes_last_active, notes_to_deactivate, t)

        # Whether the note is turned on or off, record the duration since last activity
        # Notes are not guaranteed to be toggled, e.g. a note that is already on can be turned on again
        if msg.type == 'note_on' or msg.type == 'note_off':
          last_active = notes_last_active[msg.note] # Timestep when this note was most recently activated
          if last_active != -1:
            note_duration = t - last_active
            output[last_active, msg.note, 1] = note_duration # Record the duration at the timestep it was activated
          
          if msg.type == 'note_on' and msg.velocity > 0:
            output[t, msg.note, 0] = msg.velocity / 100
            notes_last_active[msg.note] = t
            notes_is_held[msg.note] = True
          else:
            notes_is_held[msg.note] = False
            if not sustain_active:
              notes_last_active[msg.note] = -1
      still_active = (notes_last_active != -1).nonzero()[0]
      deactivate_notes(notes_last_active, still_active, t)
    return torch.from_numpy(output)

  def to_binary_velocity_tensor(tensor):
    return (tensor > 0).long()

  def calculate_compress_factor(orig_tpb, desired_tpb):
    compress_factor = orig_tpb / desired_tpb
    if compress_factor % 1.0 != 0.0:
      logging.warning(f"compress_factor of {compress_factor} is not an integer, rounding up...")
    compress_factor = math.ceil(compress_factor)
    return compress_factor

  def compress_tensor(tensor, method, orig_tpb, desired_tpb):
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
  
  def calc_dispersion(window, interval):
    deltas = [(t % interval) / interval for t in window] # Normalised distance between on-beats and note
    return circvar(deltas, high=1)

  def calc_best_interval(window, low, high):
    # Interval ticks between 32nd notes
    dispersions = np.array([MidiUtil.calc_dispersion(window, i) for i in range(low, high)])
    best = dispersions.argmin() + low
    return best

  def get_timings_in_range(note_timings, start, end):
    output = []
    for t in note_timings:
      if t >= start and t < end:
        output.append(t)
    return output
  
  def dynamic_compress_tensor(tensor, desired_tpb):
    start_interval = 40
    end_interval = 64
    w = 1920 # Window range for each slice to be processed
    min_notes_in_window = 12
    max_dispersion_diff = 0.1
    
    note_timings = torch.nonzero(torch.sum(tensor, dim=1) != 0, as_tuple=True)[0].tolist()
    start_w = 0
    prev_best_int = None
    compressed_vectors = []
    while start_w < len(tensor):
      end_w = start_w + w
      timings = MidiUtil.get_timings_in_range(note_timings, start_w, end_w)
      while len(timings) < min_notes_in_window and end_w < len(tensor):
        end_w += w
        timings = MidiUtil.get_timings_in_range(note_timings, start_w, end_w)
      best_int = MidiUtil.calc_best_interval(timings, start_interval, end_interval)
      
      if best_int != prev_best_int and prev_best_int is not None:
        best_dispersion = MidiUtil.calc_dispersion(timings, best_int)
        for i in [0, -1, 1, -2, 2]:
          alt_dispersion = MidiUtil.calc_dispersion(timings, prev_best_int + i)
          if alt_dispersion - best_dispersion < max_dispersion_diff:
            best_int = prev_best_int + i
            break
      
      tpb = best_int * 8
      compress_factor = MidiUtil.calculate_compress_factor(tpb, desired_tpb)
      for start in range(start_w, min(end_w, len(tensor)), compress_factor):
        end = start + compress_factor
        tensor_slice = tensor[start:end, :]
        compressed_vectors.append(tensor_slice.max(dim=0).values)
      start_w = end
      prev_best_int = best_int
    return torch.stack(compressed_vectors)

  def dynamic_compress_timed_tensor(tensor, desired_tpb):
    '''
      `tensor`: shape (num_timesteps, num_notes, 2), where last dimension are features (velocity, duration)
    '''
    start_interval = 40       # Minimum ticks per 1/8th beat to analyse
    end_interval = 64         # Maximum ticks per 1/8th beat to analyse
    w = 3840                  # Window range for each slice to be processed
    min_notes_in_window = 12  # Minimum number of notes within a window `w` to calculate new tpb
    max_dispersion_diff = 0.1 # Threshold between best tpb and previous window's tpb dispersions to change tpb

    # Get the timings where notes are played; multiple notes at the same timestep are only counted once
    note_timings = torch.nonzero(torch.sum(tensor[:, :, 0], dim=1) != 0, as_tuple=True)[0].tolist()
    start_w = 0
    prev_best_int = None
    compressed_vectors = []
    while start_w < len(tensor):
      end_w = start_w + w
      timings = MidiUtil.get_timings_in_range(note_timings, start_w, end_w)
      while len(timings) < min_notes_in_window and end_w < len(tensor):
        end_w += w
        timings = MidiUtil.get_timings_in_range(note_timings, start_w, end_w)
      best_int = calc_best_interval(timings, start_interval, end_interval)

      if best_int != prev_best_int and prev_best_int is not None:
        best_dispersion = calc_dispersion(timings, best_int)
        for i in [0, -1, 1, -2, 2]:
          alt_dispersion = calc_dispersion(timings, prev_best_int + i)
          if alt_dispersion - best_dispersion < max_dispersion_diff:
            best_int = prev_best_int + i
            break

      tpb = best_int * 8
      compress_factor = MidiUtil.calculate_compress_factor(tpb, desired_tpb)
      for start in range(start_w, min(end_w, len(tensor)), compress_factor):
        end = start + compress_factor
        tensor_slice = tensor[start:end]
        slice_velocities = tensor_slice[:, :, 0]
        slice_durations = tensor_slice[:, :, 1]

        # Get the highest duration per note and its corresponding velocity, in case the same note is played more than once in this window
        values, indices = slice_durations.max(dim=0)
        max_velocities = slice_velocities[indices, torch.arange(slice_velocities.shape[1])]
        max_durations = values / compress_factor
        compressed = torch.stack([max_velocities, max_durations], dim=1).float()
        compressed_vectors.append(compressed)
      start_w = end
      prev_best_int = best_int
    return torch.stack(compressed_vectors)
  
  def compress_timed_tensor(tensor, orig_tpb, desired_tpb):
    '''
    Reduces the fidelity of the musical tensor, i.e. merge multiple timesteps into one step

    Args:
      `tensor`: PyTorch tensor of shape (timesteps, num_notes)
      `orig_tpb`: original ticks per beat of the input `tensor`
      `desired_tpb`: desired ticks per beat for the tensor to be compressed to
    '''
    assert(len(tensor.shape) == 3)

    compress_factor = MidiUtil.calculate_compress_factor(orig_tpb, desired_tpb)
    compressed_vectors = []
    length = tensor.shape[0]
    for start in range(0, length, compress_factor):
      end = min(start + compress_factor, length)
      tensor_slice = tensor[start:end]
      slice_velocities = tensor_slice[:, :, 0]
      slice_durations = tensor_slice[:, :, 1]
      values, indices = slice_durations.max(dim=0)
      max_velocities = slice_velocities[indices, torch.arange(slice_velocities.shape[1])]
      max_durations = values / compress_factor
      compressed = torch.stack([max_velocities, max_durations], dim=1)
      compressed_vectors.append(compressed)
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
    compress_factor = MidiUtil.calculate_compress_factor(orig_tpb, compressed_tpb)
    # "Stretch" out the tensor using Kronecker product
    return torch.kron(tensor, torch.ones((compress_factor, 1)))

  def unreduce_tensor(tensor, start_note, end_note):
    '''
    Expands out a reduced tensor to include all 128 notes in the MIDI range

    Args:
      `tensor`: PyTorch tensor of shape (timesteps, num_notes, *)
      `start_note`: MIDI note that `tensor` starts from (integer in 0-127)
      `end_note`: MIDI note that `tensor` ends at (integer in 0-127)
    '''
    assert(end_note >= start_note)
    tensor_shape = list(tensor.shape)
    low_notes_shape = tensor_shape.copy()
    low_notes_shape[1] = start_note
    high_notes_shape = tensor_shape.copy()
    high_notes_shape[1] = 127 - end_note

    low_notes = torch.zeros(low_notes_shape).to(DEVICE)
    high_notes = torch.zeros(high_notes_shape).to(DEVICE)
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
    
  def opo_tensor_to_midi(tensor, filename, orig_tpb, compressed_tpb):
    # Create a MIDI file
    mid = mido.MidiFile()
    mid.ticks_per_beat = orig_tpb
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message('program_change', program=1))

    compress_factor = MidiUtil.calculate_compress_factor(orig_tpb, compressed_tpb)

    prev_t = 0
    for t in range(tensor.shape[0]):
      notes = torch.nonzero(tensor[t], as_tuple=True)[0].tolist()
      for i, n in enumerate(notes):
        time_delta = (t - prev_t) * compress_factor if i == 0 else 0
        new_message = mido.Message('note_on', note=n, velocity=100, time=time_delta)
        track.append(new_message)

      # Turn off the notes 1 timestep later
      for i, n in enumerate(notes):
        time_delta = compress_factor if i == 0 else 0
        new_message = mido.Message('note_on', note=n, velocity=0, time=time_delta)
        track.append(new_message)
        prev_t = t+1 # Update prev_t to t+1 since the notes are turned off 1 timestep later

    # Save the MIDI file
    mid.save(filename)
    return mid
    
  def feature_tensor_to_midi(tensor, filename, orig_tpb, compressed_tpb):
    '''
      `tensor`: shape (num_timesteps, num_notes, 2), where the last dimension are features (velocity, duration)
    '''
    mid = mido.MidiFile()
    mid.ticks_per_beat = orig_tpb
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message('program_change', program=1))

    compress_factor = MidiUtil.calculate_compress_factor(orig_tpb, compressed_tpb)

    prev_tick = 0
    notes_remaining_t = torch.zeros(tensor.shape[1]).to(device) # Number of timesteps that each note remains active/pressed
    for t in range(tensor.shape[0]):
      curr_tick = t * compress_factor

      # Check if any notes need to be turned off at this timestep
      off_notes = torch.nonzero((notes_remaining_t != 0) * (notes_remaining_t <= 1), as_tuple=True)[0].tolist()
      for n in off_notes:
        time_delta = curr_tick - prev_tick
        track.append(mido.Message('note_on', note=n, velocity=0, time=time_delta))
        prev_tick = curr_tick
      # Decrement all remaining times
      notes_remaining_t = torch.clamp(notes_remaining_t - 1, min=0)

      # Notes that are turned on at this timestep
      active_notes = torch.nonzero(tensor[t, :, 0], as_tuple=True)[0].tolist()
      for n in active_notes:
        time_delta = curr_tick - prev_tick
        # If the note was still active, turn it off before turning it on again
        if notes_remaining_t[n] > 0:
          track.append(mido.Message('note_on', note=n, velocity=0, time=time_delta))
          time_delta = 0
        velocity = int(tensor[t, n, 0] * 100)
        notes_remaining_t[n] = tensor[t, n, 1]
        track.append(mido.Message('note_on', note=n, velocity=velocity, time=time_delta))
        prev_tick = curr_tick

    # Turn off all remaining notes
    remaining_notes = torch.nonzero(notes_remaining_t, as_tuple=True)[0].tolist()
    for i, n in enumerate(remaining_notes):
      time_delta = compress_factor if i == 0 else 0
      track.append(mido.Message('note_on', note=n, velocity=0, time=time_delta))

    # Save the MIDI file
    mid.save(filename)
    return mid


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