import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

use_cuda = torch.cuda.is_available()


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
        dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    return gaussian_log_probs - torch.log(
        1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

def reparameterize(means, log_stds):
    noises = torch.randn_like(means)
    us = means + noises * log_stds.exp()
    actions = torch.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, actions)

def reparameterize_n(means, log_stds, n):
    means_n = means.reshape(means.shape[0], 1, -1).repeat(1, n, 1)
    log_stds_n = log_stds.reshape(log_stds.shape[0], 1, -1).repeat(1, n, 1)
    noises = torch.randn_like(means_n)
    us = means_n + noises * log_stds_n.exp()
    actions = torch.tanh(us)
    return actions

def reparameterize_n_clip(means, log_stds, n):
    means_n = means.reshape(means.shape[0], 1, -1).repeat(1, n, 1)
    log_stds_n = log_stds.reshape(log_stds.shape[0], 1, -1).repeat(1, n, 1)
    noises = torch.randn_like(means_n)
    us = means_n + noises * log_stds_n.exp()
    actions = torch.clip(us, -1, 1)
    return actions

def reparameterize_clip(means, log_stds):
    noises = torch.randn_like(means)
    us = means + noises * log_stds.exp()
    actions = torch.clip(us, -1, 1)
    return actions, calculate_log_pi(log_stds, noises, actions)

def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))

def evaluate_log_pi(means, log_stds, actions):
    # print("LOG_STDS", log_stds)
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)

def support_to_scalar(logits, support_size, step_size=1.0):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x * step_size for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x

def scalar_to_support(x, support_size, step_size=1.0):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = (x / step_size).floor()
    prob = (x / step_size) - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )

    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits

'''
Legacy Code:

def support_to_scalar(logits, support_size, step_size=1.0):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x * step_size for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x

def scalar_to_support(x, support_size, step_size=1.0):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size * step_size, support_size * step_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits





'''

def mlp_dropout(
        input_size,
        layer_sizes,
        output_size,
        output_activation=nn.Identity,
        momentum=0.1,
        activation=nn.ELU,
        use_bn=True,
        init_zero=False
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        layers += nn.Dropout()

        if i < len(sizes) - 2:
            act = activation
            if use_bn:
                layers += [nn.Linear(sizes[i], sizes[i + 1]),
                           nn.BatchNorm1d(sizes[i + 1], momentum=momentum),
                           act()]
            else:
                layers += [nn.Linear(sizes[i], sizes[i + 1]),
                           act()]
        else:
            act = output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       act()]
    if init_zero:
        for i in range(0, len(layers)):
            if hasattr(layers, 'weight'):
                layers[i].weight.data.fill_(0)
                layers[i].bias.data.fill_(0)

    return torch.nn.Sequential(*layers)


def mlp(
        input_size,
        layer_sizes,
        output_size,
        output_activation=nn.Identity,
        momentum=0.1,
        activation=nn.ELU,
        use_bn=True,
        init_zero=False
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            if use_bn:
                layers += [nn.Linear(sizes[i], sizes[i + 1]),
                           nn.BatchNorm1d(sizes[i + 1], momentum=momentum),
                           act()]
            else:
                layers += [nn.Linear(sizes[i], sizes[i + 1]),
                           act()]
        else:
            act = output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       act()]
    if init_zero:
        for i in range(0, len(layers)):
            if hasattr(layers, 'weight'):
                layers[i].weight.data.fill_(0)
                layers[i].bias.data.fill_(0)

    return torch.nn.Sequential(*layers)


def move_to_gpu(var):
    if use_cuda:
        return var.cuda()
    else:
        return var


def numpy_to_tensor(var):
    return move_to_gpu(Variable(torch.FloatTensor(var)))


def scalar_to_tensor(var):
    return move_to_gpu(Variable(torch.FloatTensor(np.array([var]))))


def tensor_to_scalar(t):
    return tensor_to_numpy(t).reshape(-1)[0]


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def tensor_normalize(tensor):
    if len(tensor.shape) == 2:
        return tensor / torch.norm(tensor, p=2, dim=1)
    else:
        return tensor / torch.norm(tensor, p=2, dim=0).reshape(-1)

