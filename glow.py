import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def squeeze(x):
    n, c, h, w = x.shape
    h, w = h // 2, w // 2
    x = x.reshape(n, c, h, 2, w, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    return x.reshape(n, -1, h, w)


def unsqueeze(x):
    n, _, h, w = x.shape
    x = x.reshape(n, -1, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3)
    return x.reshape(n, -1, h * 2, w * 2)


class Quantization(nn.Module):

    def forward(self, x, log_jacobian):
        return (x + torch.rand_like(x)) / 256.0 - 0.5, log_jacobian - math.log(256.0)

    def inverse(self, z, log_jacobian):
        return torch.clamp((z + 0.5) * 256.0, 0, 255), log_jacobian + math.log(256.0)


class ActNorm(nn.Module):

    def __init__(self, channels, scale=1.0, logscale_factor=3.0):
        super(ActNorm, self).__init__()

        self.scale, self.logscale_factor = scale, logscale_factor

        self.mean = nn.Parameter(torch.zeros(1, channels, 1, 1, requires_grad=True))
        self.logstd  = nn.Parameter(torch.zeros(1, channels, 1, 1, requires_grad=True))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize_mean_and_logstd(self, x):
        if self.initialized.item() == 0:
            mean = torch.mean(x, (0, 2, 3), keepdim=True).detach()
            std  = torch.std(x, (0, 2, 3), keepdim=True).detach()
            self.mean.data = -mean
            self.logstd.data  = torch.log(self.scale / (std + 1e-6)) / self.logscale_factor
            self.initialized.fill_(1)

    def forward(self, x):
        self.initialize_mean_and_logstd(x)
        logstd = self.logstd * self.logscale_factor
        return (x + self.mean) * torch.exp(logstd)


class ActNormFlow(ActNorm):

    def forward(self, x, log_jacobian):
        self.initialize_mean_and_logstd(x)
        logstd = self.logstd * self.logscale_factor
        return (x + self.mean) * torch.exp(logstd), log_jacobian + logstd

    def inverse(self, z, log_jacobian):
        logstd = self.logstd * self.logscale_factor
        return z * torch.exp(-logstd) - self.mean, log_jacobian - logstd


class CouplingFlow(nn.Module):

    def __init__(self, f, eps=2.0):
        super(CouplingFlow, self).__init__()

        self.f = f
        self.eps = eps

    def _get_scale_and_shift(self, x):
        y = self.f(x)
        return torch.sigmoid(y[:, 1::2, :, :] + self.eps), y[:, 0::2, :, :]

    def forward(self, x, log_jacobian):
        x0, x1 = x.chunk(2, 1)
        scale, shift = self._get_scale_and_shift(x0)
        z = torch.cat((x0, (x1 + shift) * scale), 1)
        log_jacobian = log_jacobian + torch.cat((torch.zeros_like(scale), torch.log(scale)), 1)
        return z, log_jacobian

    def inverse(self, z, log_jacobian):
        z0, z1 = z.chunk(2, 1)
        scale, shift = self._get_scale_and_shift(z0)
        x = torch.cat((z0, (z1 / scale) - shift), 1)
        log_jacobian = log_jacobian - torch.cat((torch.zeros_like(scale), torch.log(scale)), 1)
        return x, log_jacobian


class InvConv2dFlow(nn.Module):

    def __init__(self, dim):
        super(InvConv2dFlow, self).__init__()

        self.w = nn.Parameter(torch.qr(torch.randn(dim, dim))[0])

    def forward(self, x, log_jacobian):
        z = F.conv2d(x, self.w.unsqueeze(-1).unsqueeze(-1))
        log_jacobian = log_jacobian + (torch.slogdet(self.w)[1] / x.size(1))
        return z, log_jacobian

    def inverse(self, z, log_jacobian):
        w = torch.inverse(self.w)
        x = F.conv2d(z, w.unsqueeze(-1).unsqueeze(-1))
        log_jacobian = log_jacobian + (torch.slogdet(w)[1] / z.size(1))
        return x, log_jacobian


class ChainFlow(nn.Module):

    def __init__(self, flows):
        super(ChainFlow, self).__init__()

        self.flows = nn.ModuleList(flows)

    def forward(self, x, log_jacobian):
        for flow in self.flows:
            x, log_jacobian = flow(x, log_jacobian)
        return x, log_jacobian

    def inverse(self, z, log_jacobian):
        for flow in reversed(self.flows):
            z, log_jacobian = flow.inverse(z, log_jacobian)
        return z, log_jacobian


class Conv2dZeros(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, logscale_factor=3.0):
        super(Conv2dZeros, self).__init__()

        self.logscale_factor = logscale_factor

        self.scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)

        nn.init.constant_(self.conv.weight, 0.0)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        return self.conv(x) * torch.exp(self.scale * self.logscale_factor)


def gaussian_log_prob(x, mean, logstd):
    return -.5 * (math.log(2.0 * math.pi) + 2.0 * logstd + (x - mean) ** 2 / torch.exp(2.0 * logstd))

class Split(nn.Module):

    def __init__(self, f):
        super(Split, self).__init__()

        self.f = f

    def _get_mean_and_logstd(self, x):
        y = self.f(x)
        return y[:, 0::2, :, :], y[:, 1::2, :, :]

    def forward(self, x, log_jacobian):
        x0, x1 = x.chunk(2, 1)
        log_jacobian0, log_jacobian1 = log_jacobian.chunk(2, 1)
        mean, logstd = self._get_mean_and_logstd(x0)
        log_prob = torch.sum(gaussian_log_prob(x1, mean, logstd) + log_jacobian1, dim=(1, 2, 3))
        return x0, log_jacobian0, log_prob

    def sample(self, z0, log_jacobian0, eps):
        mean, logstd = self._get_mean_and_logstd(z0)
        z1 = mean + torch.exp(logstd) * eps
        return torch.cat((z0, z1), 1), torch.cat((log_jacobian0, torch.zeros_like(log_jacobian0)), 1)

class Prior(Split):

    def forward(self, x, log_jacobian):
        mean, logstd = self._get_mean_and_logstd(torch.zeros_like(x))
        log_prob = torch.sum(gaussian_log_prob(x, mean, logstd) + log_jacobian, dim=(1, 2, 3))
        return None, None, log_prob

    def sample(self, z0, log_jacobian0, eps):
        assert z0 is None and log_jacobian0 is None
        mean, logstd = self._get_mean_and_logstd(torch.zeros_like(eps))
        z = mean + torch.exp(logstd) * eps
        return z, torch.zeros_like(z)


def f(in_channels, out_channels, hidden_channels):
    model = nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
        ActNorm(hidden_channels),
        nn.ReLU(),
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
        ActNorm(hidden_channels),
        nn.ReLU(),
        Conv2dZeros(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
    )
    nn.init.normal_(model[0].weight, 0.0, 0.05)
    nn.init.normal_(model[3].weight, 0.0, 0.05)
    return model


class Glow(nn.Module):

    def __init__(self, flows, priors, eps_shapes):
        super(Glow, self).__init__()

        self.flows = nn.ModuleList(flows)
        self.priors = nn.ModuleList(priors)
        self.eps_shapes = eps_shapes

    def forward(self, x):
        log_prob = torch.zeros(x.size(0), dtype=torch.float32, device=x.device)
        log_jacobian = torch.zeros_like(x)
        for flow, prior in zip(self.flows, self.priors):
            x, log_jacobian = squeeze(x), squeeze(log_jacobian)
            x, log_jacobian = flow(x, log_jacobian)
            x, log_jacobian, partial_log_prob = prior(x, log_jacobian)
            log_prob += partial_log_prob
        assert x is None and log_jacobian is None
        return log_prob

    def sample(self, epss):
        flows, priors, epss = reversed(self.flows), reversed(self.priors), reversed(epss)
        z, log_jacobian = None, None
        for flow, prior, eps in zip(flows, priors, epss):
            z, log_jacobian = prior.sample(z, log_jacobian, eps)
            z, log_jacobian = flow.inverse(z, log_jacobian)
            z, log_jacobian = unsqueeze(z), unsqueeze(log_jacobian)
        return z, log_jacobian

def glow(image_size, in_channels, n_levels, depth, hidden_channels):
    flows, priors, eps_shapes = [], [], []
    for level in range(n_levels):
        in_channels, image_size = in_channels * 4, image_size // 2

        level_flows = []
        if level == 0:
            level_flows.append(Quantization())

        for _ in range(depth):
            level_flows.extend([
                ActNormFlow(in_channels),
                InvConv2dFlow(in_channels),
                CouplingFlow(f(in_channels // 2, in_channels, hidden_channels))
            ])
        flows.append(ChainFlow(level_flows))
        if level == n_levels - 1:
            priors.append(Prior(Conv2dZeros(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1)))
            eps_shapes.append((in_channels, image_size, image_size))
        else:
            priors.append(Split(Conv2dZeros(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1)))
            eps_shapes.append((in_channels // 2, image_size, image_size))
        in_channels //= 2
    return Glow(flows, priors, eps_shapes)
