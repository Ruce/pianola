import json
import logging
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, ReLU, Module, Sequential
from torch_geometric.nn import MessagePassing

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

###########################################################
# ---------- Scatter function from torch-scatter ----------
# Using raw Python functions as torch-scatter fails to build properly in Sagemaker
from typing import Optional, Tuple

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mul(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.scatter_mul(src, index, dim, out, dim_size)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out


def scatter_min(
        src: torch.Tensor, index: torch.Tensor, dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_min(src, index, dim, out, dim_size)


def scatter_max(
        src: torch.Tensor, index: torch.Tensor, dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_max(src, index, dim, out, dim_size)


def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
            out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor:
    r"""
    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/add.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Reduces all values from the :attr:`src` tensor into :attr:`out` at the
    indices specified in the :attr:`index` tensor along a given axis
    :attr:`dim`.
    For each value in :attr:`src`, its output index is specified by its index
    in :attr:`src` for dimensions outside of :attr:`dim` and by the
    corresponding value in :attr:`index` for dimension :attr:`dim`.
    The applied reduction is defined via the :attr:`reduce` argument.

    Formally, if :attr:`src` and :attr:`index` are :math:`n`-dimensional
    tensors with size :math:`(x_0, ..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})`
    and :attr:`dim` = `i`, then :attr:`out` must be an :math:`n`-dimensional
    tensor with size :math:`(x_0, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})`.
    Moreover, the values of :attr:`index` must be between :math:`0` and
    :math:`y - 1`, although no specific ordering of indices is required.
    The :attr:`index` tensor supports broadcasting in case its dimensions do
    not match with :attr:`src`.

    For one-dimensional tensors with :obj:`reduce="sum"`, the operation
    computes

    .. math::
        \mathrm{out}_i = \mathrm{out}_i + \sum_j~\mathrm{src}_j

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    .. note::

        This operation is implemented via atomic operations on the GPU and is
        therefore **non-deterministic** since the order of parallel operations
        to the same value is undetermined.
        For floating-point variables, this results in a source of variance in
        the result.

    :param src: The source tensor.
    :param index: The indices of elements to scatter.
    :param dim: The axis along which to index. (default: :obj:`-1`)
    :param out: The destination tensor.
    :param dim_size: If :attr:`out` is not given, automatically create output
        with size :attr:`dim_size` at dimension :attr:`dim`.
        If :attr:`dim_size` is not given, a minimal sized output tensor
        according to :obj:`index.max() + 1` is returned.
    :param reduce: The reduce operation (:obj:`"sum"`, :obj:`"mul"`,
        :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`). (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`

    .. code-block:: python

        from torch_scatter import scatter

        src = torch.randn(10, 6, 64)
        index = torch.tensor([0, 1, 0, 1, 2, 1])

        # Broadcasting in the first and last dim.
        out = scatter(src, index, dim=1, reduce="sum")

        print(out.size())

    .. code-block::

        torch.Size([10, 3, 64])
    """
    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == 'mul':
        return scatter_mul(src, index, dim, out, dim_size)
    elif reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
    elif reduce == 'min':
        return scatter_min(src, index, dim, out, dim_size)[0]
    elif reduce == 'max':
        return scatter_max(src, index, dim, out, dim_size)[0]
    else:
        raise ValueError

# ---------------- End of scatter function ----------------
###########################################################

class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=6, aggr='add'):
        """Message Passing Neural Network Layer

        Args:
            emb_dim: (int) - hidden dimension `d`
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.mlp_msg = Sequential(Linear(2*emb_dim + edge_dim, emb_dim), ReLU(), Linear(emb_dim, emb_dim), ReLU())
        self.mlp_upd = Sequential(Linear(2*emb_dim, emb_dim), ReLU(), Linear(emb_dim, emb_dim), ReLU())

    def forward(self, h, edge_index, edge_attr):
        """
        Args:
            h: (n, d) - initial node features
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_attr: (e, d_e) - edge features

        Returns:
            out: (n, d) - updated node features
        """
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        """
        Args:
            h_i: (e, d) - destination node features, essentially h[edge_index[0]]
            h_j: (e, d) - source node features, essentially h[edge_index[1]]

        Returns:
            msg: (e, d) - messages `m_ij` passed through MLP `\psi`
        """
        if len(h_i.shape) == 3:
            # First dimension is batch size, repeat (i.e. tile) edge_attr
            edge_attr = edge_attr.repeat(h_i.shape[0], 1, 1)
        elif len(h_i.shape) == 4:
            edge_attr = edge_attr.repeat(h_i.shape[1], 1, 1).repeat(h_i.shape[0], 1, 1, 1)

        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)

    def aggregate(self, inputs, index):
        """
        Args:
            inputs: (e, d) - messages `m_ij` from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: (n, d) - aggregated messages `m_i`
        """
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def update(self, aggr_out, h):
        """
        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
        """
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')

class MPNNModel(nn.Module):
    def __init__(self, edge_index, edge_attr, in_dim, num_layers=4, emb_dim=64, out_dim=1):
        """Message Passing Neural Network model for graph property prediction

        Args:
            in_dim: (int) - initial node feature dimension `d_n`
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()

        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.edge_dim = self.edge_attr.shape[1]

        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in1 = Linear(in_dim, emb_dim)

        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, self.edge_dim, aggr='add'))

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred1 = Linear(emb_dim, out_dim)

    def forward(self, x):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns:
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in1(x)
        for conv in self.convs:
            h = h + conv(h, self.edge_index, self.edge_attr)
        out = self.lin_pred1(h)
        out = torch.sum(out, dim=-2)
        return out

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

class SequenceModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_notes):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_notes)

    def forward(self, x, h_0=None, c_0=None):
        if h_0 is None or c_0 is None:
            lstm_out, (h_n, c_n) = self.lstm(x)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        y_hat = self.linear(lstm_out)
        return y_hat, (h_n, c_n)
        
class ConvSequence(nn.Module):
    def __init__(self, num_notes, first_channel, second_channel, conv_emb_dim, conv_out_dim, seq_emb_dim):
        super().__init__()
        self.conv_model = ConvModel(num_notes, first_channel, second_channel, conv_emb_dim, conv_out_dim)
        self.sequence_model = SequenceModel(conv_out_dim, seq_emb_dim, num_notes)

    def forward(self, x, h_0=None, c_0=None):
        h = self.conv_model(x)
        out, (h_n, c_n) = self.sequence_model(h, h_0, c_0)
        return out, (h_n, c_n)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_NOTES = 64
WINDOW_SIZE = 64
NEIGHBOUR_DISTANCES = [-7, -4, -3, 3, 4, 7]
NOTES_DELIMITER = ','
TICKS_DELIMITER = ';'

def create_tonnetz_adjacency_matrix(num_notes):
    # In Tonnetz, each node has six neighbours which have pitches of the following distances (in semi-tones)
    # E.g. C4 has neighbours F3, G#3, A3, D#4, E4, G4
    A = []
    for i in range(num_notes):
        row = torch.zeros(num_notes, dtype=torch.int)
        for d in NEIGHBOUR_DISTANCES:
            j = i+d
            if j >= 0 and j < num_notes:
                row[j] = 1
        A.append(row)
    A = torch.stack(A)
    # Check that A is symmetric since the Tonnetz graph is undirected
    assert(torch.equal(A, A.transpose(0, 1)))

    # Convert to sparse format expected by PyG layers
    edge_index = A.to_sparse().indices().to(device)
    return edge_index

def create_tonnetz_edge_attr(edge_index):
    edge_attr_indices = []
    for i in range(edge_index.shape[1]):
        distance = (edge_index[1][i] - edge_index[0][i]).item()
        edge_attr_indices.append(NEIGHBOUR_DISTANCES.index(distance))

    edge_attr = F.one_hot(torch.tensor(edge_attr_indices)).to(device)
    return edge_attr

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
    # For conv-sequence model, expected input shape is (batch_size, timesteps, num_notes, 1)
    generated = seed.unsqueeze(dim=0).unsqueeze(dim=-1)
    window_size = generated.shape[1]

    model.eval()
    with torch.no_grad():
        h_n, c_n = (None, None)
        for i in range(timesteps):
            if i == 0:
                pred, (h_n, c_n) = model(generated[:, -window_size:])
            else:
                pred, (h_n, c_n) = model(generated[:, -1:], h_n, c_n)
            y_hat = torch.sigmoid(pred[:, -1]).squeeze(dim=0) # Keep only the last timestep and remove batch_size dimension
            new_notes = decode_tensor(y_hat, max_notes) # Decode probabilities in y_hat
            generated = torch.cat((generated, new_notes.reshape((1, 1, -1, 1))), dim=1)
    return generated[0, -timesteps:].squeeze(dim=-1) # Remove batch_size and node_features dimensions

# defining model and loading weights to it.
def model_fn(model_dir):
    #edge_index = create_tonnetz_adjacency_matrix(NUM_NOTES)
    #edge_attr = create_tonnetz_edge_attr(edge_index)
    model = ConvSequence(num_notes=64, first_channel=8, second_channel=32, conv_emb_dim=512, conv_out_dim=128, seq_emb_dim=512)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))
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
    # Pad or trim input_object['data'] tensor to the correct WINDOW_SIZE
    padding = torch.zeros((WINDOW_SIZE, NUM_NOTES))
    seed = torch.cat((padding, input_object['data']), dim=0)[-WINDOW_SIZE:]
    prediction = generate_music(model, seed, timesteps)
    return prediction


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    res = notes_tensor_to_str(predictions)
    return json.dumps(res)