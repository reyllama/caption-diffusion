import torch
from torch.nn import functional as F

def d_clip_loss(x, y, use_cosine=False):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    if use_cosine:
        distance = 1 - (x @ y.t()).squeeze()
    else:
        distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    return distance


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

def calc_style_loss(gen, style):
    batch_size, channel, height, width = gen.shape

    G = torch.mm(gen.view(channel, height * width), gen.view(channel, height * width).t())
    A = torch.mm(style.view(channel, height * width), style.view(channel, height * width).t())

    style_l = torch.mean((G - A) ** 2)
    return style_l

def calculate_loss(gen_features, style_featues):
    style_loss = 0
    for gen, style in zip(gen_features, style_featues):
        # extracting the dimensions from the generated image
        style_loss += calc_style_loss(gen, style)

    # calculating the total loss of e th epoch
    return style_loss