import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import init
import geoopt.manifolds.poincare.math as pmath
from utils.train_utils import glorot, zeros, add_self_loops
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, softmax
import manifolds
import scipy.sparse as sp
import numpy as np


def init_weight(modules, activation):
    """
    Weight initialization
    :param modules: Iterable of modules
    :param activation: Activation function.
    """
    for m in modules:
        if isinstance(m, nn.Linear):
            if activation is None:
                m.weight.data = init.xavier_uniform_(
                    m.weight.data)
            else:
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain(activation.lower()))
            if m.bias is not None:
                m.bias.data = init.constant_(m.bias.data, 0.0)


def get_act_function(activate_function):
    """
    Get activation function by name
    :param activation_fuction: Name of activation function
    """
    if activate_function == 'sigmoid':
        activate_function = nn.Sigmoid()
    elif activate_function == 'relu':
        activate_function = nn.ReLU()
    elif activate_function == 'tanh':
        activate_function = nn.Tanh()
    else:
        return None
    return activate_function


class CombineModel(nn.Module):
    def __init__(self):
        super(CombineModel, self).__init__()
        self.thetas = nn.Parameter(torch.ones(3))

    def loss(self, S1, S2, S3, id2idx_augment):
        S = self.forward(S1, S2, S3)
        S_temp = torch.zeros(S.shape)
        for k, v in id2idx_augment.items():
            S_temp[int(k), v] = 1

        S = S / torch.sqrt((S ** 2).sum(dim=1)).view(S.shape[0], 1)
        loss = -(S * S_temp).mean()
        return loss

    def forward(self, S1, S2, S3):
        theta_sum = torch.abs(self.thetas[0]) + torch.abs(self.thetas[1]) + torch.abs(self.thetas[2])
        return (torch.abs(self.thetas[0]) / theta_sum) * torch.Tensor(S1) + (torch.abs(self.thetas[1]) / theta_sum) * torch.Tensor(S2) + (torch.abs(self.thetas[2]) / theta_sum) * torch.Tensor(S3)

class Combine4Model(nn.Module):
    def __init__(self):
        super(Combine4Model, self).__init__()
        self.thetas = nn.Parameter(torch.ones(4))

    def loss(self, S1, S2, S3, S4, id2idx_augment):
        S = self.forward(S1, S2, S3, S4)
        S_temp = torch.zeros(S.shape)
        for k, v in id2idx_augment.items():
            S_temp[int(k), v] = 1

        S = S / torch.sqrt((S ** 2).sum(dim=1)).view(S.shape[0], 1)
        loss = -(S * S_temp).mean()
        return loss

    def forward(self, S1, S2, S3, S4):
        theta_sum = torch.abs(self.thetas[0]) + torch.abs(self.thetas[1]) + torch.abs(self.thetas[2]) + torch.abs(self.thetas[3])
        return (torch.abs(self.thetas[0]) / theta_sum) * torch.Tensor(S1) + (torch.abs(self.thetas[1]) / theta_sum) * torch.Tensor(S2) + (
                    torch.abs(self.thetas[2]) / theta_sum) * torch.Tensor(S3) + (torch.abs(self.thetas[3]) / theta_sum) * torch.Tensor(S4)


class Combine2Model(nn.Module):
    def __init__(self):
        super(Combine2Model, self).__init__()
        self.thetas = nn.Parameter(torch.ones(2))

    def loss(self, S1, S2, id2idx_augment):
        S = self.forward(S1, S2)
        S_temp = torch.zeros(S.shape)
        for k, v in id2idx_augment.items():
            S_temp[int(k), v] = 1

        S = S / torch.max(S, dim=1)[0].view(S.shape[0], 1)
        loss = -(S * S_temp).mean()
        return loss

    def forward(self, S1, S2):
        return torch.abs(self.thetas[0]) * torch.Tensor(S1) + torch.abs(self.thetas[1]) * torch.Tensor(S2)


class GCN(nn.Module):
    """
    The GCN multistates block
    """

    def __init__(self, activate_function, input_dim, output_dim):
        """
        activate_function: Tanh
        input_dim: input features dimensions
        output_dim: output features dimensions
        """
        super(GCN, self).__init__()
        if activate_function is not None:
            self.activate_function = get_act_function(activate_function)
        else:
            self.activate_function = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        init_weight(self.modules(), activate_function)

    def forward(self, input, A_hat):
        output = self.linear(input)
        output = torch.matmul(A_hat, output)
        if self.activate_function is not None:
            output = self.activate_function(output)
        return output

def create_wmlp(widths,nfeato,lbias):
    mlp_modules=[]
    for k in range(len(widths)-1):
        mlp_modules.append(nn.Linear(widths[k],widths[k+1],bias=False))
        mlp_modules.append(nn.LeakyReLU(0.8,True))
    mlp_modules.append(nn.Linear(widths[len(widths)-1],nfeato,bias=lbias))
    return nn.Sequential(*mlp_modules)

class GILayer(nn.Module):
    def __init__(self, activate_function, input_dim, output_dim):
        super(GILayer, self).__init__()
        self.manifold = getattr(manifolds, 'PoincareBall')()
        if activate_function is not None:
            self.activate_function = get_act_function(activate_function)
        else:
            self.activate_function = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        init_weight(self.modules(), activate_function)
        self.conv = HGATConv(self.manifold, input_dim, output_dim, 8, 1, 0.2, 0.0,
                             1, self.activate_function, atten=1, dist=1)
        self.conv_e = GATConv(input_dim, output_dim, 8, 1, 0.2,
                              0.0, 1, self.activate_function)

    def forward(self, input, A_hat, w_mul):
        x_e, x = input
        adj = A_hat
        "hyper forward"
        x = self.conv(x, adj, w_mul)
        "eucl forward"
        x_e = self.conv_e(x_e, adj, w_mul)
        return (x_e, x), adj, w_mul


class HGATConv(MessagePassing):
    def __init__(self,
                 manifold,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 act=None,
                 atten=True,
                 dist=True):
        super(HGATConv, self).__init__('add')

        self.manifold = manifold
        self.c = 1.0
        self.concat = concat
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if concat:
            self.out_channels = out_channels // heads
        else:
            self.out_channels = out_channels

        self.in_channels = in_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.act = act
        self.dist = dist
        self.atten = atten

        self.hy_linear = HypLinear(manifold, in_channels, heads * self.out_channels, 1, dropout, bias)
        self.att = Parameter(torch.Tensor(1, heads, 2 * self.out_channels))

        widths = [1, out_channels]
        self.w_mlp_out = create_wmlp(widths, 8, 1)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)
        self.hy_linear.reset_parameters()

    def forward(self, x, adj, w_mul):
        x = self.hy_linear.forward(x)
        w_mul = w_mul.reshape(-1, 1)
        out_weight = self.w_mlp_out(w_mul)
        edge_index = adj._indices()
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        log_x = pmath.logmap0(x, c=1.0)
        log_x = log_x.view(-1, self.heads, self.out_channels)
        out_weight = softmax(out_weight, edge_index[0])
        out = self.propagate(edge_index, x=log_x, num_nodes=x.size(0), original_x=x, w_mul=out_weight)
        out = self.manifold.proj_tan0(out, c=self.c)

        out = self.act(out)
        out = self.manifold.proj_tan0(out, c=self.c)

        return self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)

    def message(self, edge_index_i, x_i, x_j, num_nodes, original_x_i, original_x_j, w_mul):
        # Compute attention coefficients.
        if self.atten:
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            if self.dist:  # Compute distance
                dist = pmath.dist(original_x_i, original_x_j)
                dist = softmax(dist, edge_index_i, num_nodes).reshape(-1, 1)
                #
                alpha = alpha * dist * w_mul
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, edge_index_i, num_nodes)
            # Sample attention coefficients stochastically.
            if self.training and self.dropout > 0:
                alpha = F.dropout(alpha, p=self.dropout, training=True)

            return x_j * alpha.view(-1, self.heads, 1)
        else:
            return x_j

    def update(self, aggr_out):
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias, self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )

class GATConv(MessagePassing):
    """The graph attentional operator from the "Graph Attention Networks"
    Implementation based on torch_geometric
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 act=None):
        super(GATConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.act = act
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if concat:
            self.out_channels = out_channels // heads
        else:
            self.out_channels = out_channels

        self.linear = torch.nn.Linear(in_channels, heads * self.out_channels, bias=bias)
        self.att = Parameter(torch.Tensor(1, heads, 2 * self.out_channels))

        widths = [1, out_channels]
        self.w_mlp_out = create_wmlp(widths, 8, 1)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)
        self.linear.reset_parameters()

    def forward(self,  x, adj, w_mul):
        w_mul = w_mul.reshape(-1, 1)
        out_weight = self.w_mlp_out(w_mul)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        edge_index = adj._indices()
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.linear(x).view(-1, self.heads, self.out_channels)
        out_weight = softmax(out_weight, edge_index[0])
        out = self.propagate(edge_index, x=x, num_nodes=x.size(0), w_mul=out_weight)
        out = self.act(out)
        return out

    def message(self, edge_index_i, x_i, x_j, num_nodes, w_mul):
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = alpha * w_mul
        alpha = F.leaky_relu(alpha , self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes)
        # Sample attention coefficients stochastically.
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class CINA_MODEL(nn.Module):
    """
    Training a multilayer GCN model
    """

    def __init__(self, activate_function, num_GCN_blocks, input_dim, output_dim, \
                 num_source_nodes, num_target_nodes, source_info, target_info, source_feats=None, target_feats=None):
        """
        :params activation_fuction: Name of activation function
        :params num_GCN_blocks: Number of GCN layers of model
        :params input_dim: The number of dimensions of input
        :params output_dim: The number of dimensions of output
        :params num_source_nodes: Number of nodes in source graph
        :params num_target_nodes: Number of nodes in target graph
        :params source_feats: Source Initialized Features
        :params target_feats: Target Initialized Features
        """
        super(CINA_MODEL, self).__init__()
        self.num_GCN_blocks = num_GCN_blocks
        self.source_feats = source_feats
        self.target_feats = target_feats
        self.c = torch.tensor([1.])
        self.manifold = getattr(manifolds, 'PoincareBall')()
        input_dim = self.source_feats.shape[1]
        self.input_dim = input_dim
        self.source_info = source_info
        self.target_info = target_info

        # GCN blocks (emb)
        self.GCNs = []
        for i in range(num_GCN_blocks):
            self.GCNs.append(GILayer(activate_function, input_dim, output_dim))
            input_dim = self.GCNs[-1].output_dim
        self.GCNs = nn.ModuleList(self.GCNs)
        init_weight(self.modules(), activate_function)

    def forward(self, A_hat, w_mul, net):
        """
        Do the forward
        :params A_hat: The sparse Normalized Laplacian Matrix
        :params net: Whether forwarding graph is source or target graph
        """
        if net == 's':
            input_x_e = self.source_feats
        else:
            input_x_e = self.target_feats

        input_x_h = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(input_x_e, c=self.c), c=self.c),
            c=self.c)

        input = (input_x_e, input_x_h)
        emb_input = (input_x_e, input_x_h)
        outputs = [emb_input]
        for i in range(self.num_GCN_blocks):
            GCN_output_i = self.GCNs[i](emb_input, A_hat, w_mul)
            outputs.append(GCN_output_i[0][0:2])
            emb_input = GCN_output_i[0][0:2]
        return outputs

class HCINA_MODEL(nn.Module):

    def __init__(self, embedding_dim, source_feats, target_feats):
        super(HCINA_MODEL, self).__init__()
        self.source_feats = source_feats
        self.target_feats = target_feats
        self.maps = nn.Linear(source_feats.shape[1], embedding_dim, bias=True)

    def forward(self, feats, hg, v2e_weight, e2v_weight):
        """
        Do the forward
        :params A_hat: The sparse Normalized Laplacian Matrix
        :params net: Whether forwarding graph is source or target graph
        """
        output = self.maps(feats)
        Y = hg.v2e(output, v2e_weight=v2e_weight, aggr="mean")
        X = hg.e2v(Y, e2v_weight=e2v_weight, aggr="mean")

        Y = hg.v2e(X, v2e_weight=v2e_weight, aggr="mean")
        X = hg.e2v(Y, e2v_weight=e2v_weight, aggr="mean")
        X = torch.softmax(X, dim=1)
        return X

class StableFactor(nn.Module):
    """
    Stable factor following each node
    """

    def __init__(self, num_source_nodes, num_target_nodes, cuda=True):
        """
        :param num_source_nodes: Number of nodes in source graph
        :param num_target_nodes: Number of nodes in target graph
        """
        super(StableFactor, self).__init__()
        self.alpha_source_e = torch.ones(num_source_nodes)
        self.alpha_target_e = torch.ones(num_target_nodes)
        self.alpha_source_h = torch.ones(num_source_nodes)
        self.alpha_target_h = torch.ones(num_target_nodes)
        self.score_max_e = 0
        self.score_max_h = 0
        self.alpha_source_max_e = None
        self.alpha_target_max_e = None
        self.alpha_source_max_h = None
        self.alpha_target_max_h = None
        if cuda:
            self.alpha_source_e = self.alpha_source_e.cuda()
            self.alpha_target_e = self.alpha_target_e.cuda()
            self.alpha_source_h = self.alpha_source_h.cuda()
            self.alpha_target_h = self.alpha_target_h.cuda()
        self.use_cuda = cuda

    def forward(self, A_hat, net='s'):
        """
        Do the forward
        :param A_hat is the Normalized Laplacian Matrix
        :net: whether graph considering is source or target graph.
        """
        if net == 's':
            self.alpha = self.alpha_source_e
        else:
            self.alpha = self.alpha_target_e
        alpha_colum = self.alpha.reshape(len(self.alpha), 1)
        if self.use_cuda:
            alpha_colum = alpha_colum.cuda()
        A_hat = A_hat.to_dense()
        A_hat_new = (alpha_colum * (A_hat * alpha_colum).t()).t()
        tmp_coo = sp.coo_matrix(A_hat_new.cpu())
        values = tmp_coo.data
        indices = np.vstack((tmp_coo.row, tmp_coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        A_hat_new = torch.sparse_coo_tensor(i, v, tmp_coo.shape)
        return A_hat_new


