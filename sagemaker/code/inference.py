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

        # MLP `\psi` for computing messages `m_ij`
        # dims: (2d + d_e) -> d
        #self.mlp_msg = Sequential(
        #    Linear(2*emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
        #    Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        #  )
        self.mlp_msg = Sequential(Linear(2*emb_dim + edge_dim, emb_dim), ReLU(), Linear(emb_dim, emb_dim), ReLU())

        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        # dims: 2d -> d
        #self.mlp_upd = Sequential(
        #    Linear(2*emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
        #    Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        #  )
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
    def __init__(self, edge_index, edge_attr, in_dim, num_layers, emb_dim, out_dim=1):
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
        self.lin_in2 = Linear(emb_dim, emb_dim)

        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, self.edge_dim, aggr='add'))

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred1 = Linear(emb_dim, emb_dim)
        self.lin_pred2 = Linear(emb_dim, out_dim)

    def forward(self, x):
        """
        Args:
          data: (PyG.Data) - batch of PyG graphs

        Returns:
          out: (batch_size, out_dim) - prediction for each graph
        """
        h = F.relu(self.lin_in1(x))
        h = F.relu(self.lin_in2(h))

        for conv in self.convs:
            h = conv(h, self.edge_index, self.edge_attr)

        h = F.relu(self.lin_pred1(h))
        out = self.lin_pred2(h)
        return out.squeeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_NOTES = 64
WINDOW_SIZE = 48
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
    ticks = notes_tensor.shape[1]

    notes_list = []
    for t in range(ticks):
        active_notes_in_slice = torch.nonzero(notes_tensor[:, t], as_tuple=True)[0]
        notes_list.append(NOTES_DELIMITER.join([str(n) for n in active_notes_in_slice.tolist()])) # Convert tensor to list to string
    return TICKS_DELIMITER.join(notes_list)

def notes_str_to_tensor(notes_str, num_notes):
    notes_slices = notes_str.split(TICKS_DELIMITER)
    notes_tensor = torch.zeros((num_notes, len(notes_slices)))

    for t, active_notes_str in enumerate(notes_slices):
        active_notes = [int(n) for n in active_notes_str.split(NOTES_DELIMITER)]
        notes_tensor[active_notes, t] = 1
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
    generated = seed
    window_size = seed.shape[1]

    model.eval()
    with torch.no_grad():
        for i in range(timesteps):
            if i == 0:
                pred = model(seed)
            else:
                pred = model(generated[:, -window_size:])

            y_hat = torch.sigmoid(pred)
            new_notes = decode_tensor(y_hat, max_notes)
            generated = torch.cat((generated, new_notes.unsqueeze(1)), dim=1)
    return generated[:, -timesteps:]


# defining model and loading weights to it.
def model_fn(model_dir):
    edge_index = create_tonnetz_adjacency_matrix(NUM_NOTES)
    edge_attr = create_tonnetz_edge_attr(edge_index)
    model = MPNNModel(edge_index, edge_attr, in_dim=WINDOW_SIZE, num_layers=4, emb_dim=48, out_dim=1)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))
    model.to(device).eval()
    return model


# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    notes_str = json.loads(request_body)["inputs"]
    data = notes_str_to_tensor(notes_str, num_notes=NUM_NOTES)
    return data


# inference
def predict_fn(input_object, model):
    timesteps = 16
    # Pad or trim input_object tensor to the correct WINDOW_SIZE
    seed = torch.cat((torch.zeros((NUM_NOTES, WINDOW_SIZE)), input_object), dim=1)[:, -WINDOW_SIZE:]
    prediction = generate_music(model, seed, timesteps)
    return prediction


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    res = notes_tensor_to_str(predictions)
    return json.dumps(res)