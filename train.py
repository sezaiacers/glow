import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

import argparse
import time
import os
import math

import glow

def generate_batch_eps(eps_shapes, stds, n, batch_size):
    current_size, batch_eps = 0, []
    while current_size < n:
        batch_eps.append([torch.randn(batch_size, *shape) * std for shape, std  in zip(eps_shapes, stds)])
        current_size += batch_size
    return batch_eps

def main(args):
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = glow.glow(args.image_size, args.in_channels, args.n_levels, args.depth, args.hidden_channels).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.images, exist_ok=True)
    batch_eps = generate_batch_eps(model.eps_shapes, [0.01, 0.125, 0.25, 0.5], 256, args.batch_size)
    def generate(epoch, global_step):
        model.eval()
        xs = []
        with torch.no_grad():
            for eps in batch_eps:
                xs.append(model.sample([e.to(device=device) for e in eps])[0])
        x = torch.cat(xs, 0)[:256] / 255.0
        torchvision.utils.save_image(x.cpu(), os.path.join(args.images, '{}_{}.jpg'.format(epoch, global_step)), nrow=16)

    os.makedirs(args.model_dir, exist_ok=True)
    def save(epoch, global_step):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': global_step
        }
        torch.save(checkpoint, os.path.join(args.model_dir, '{}_{}.pth.tar'.format(epoch, global_step)))

    start_epoch, global_step = 0, 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch, global_step = checkpoint['epoch'], checkpoint['global_step'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    def train(epoch, global_step):
        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        progress = ProgressMeter(len(train_dataloader), [batch_time, losses], prefix="Epoch: [{}]".format(epoch))

        model.train()
        end = time.time()
        for i, (x, _) in enumerate(train_dataloader):
            x = x.to(device=device).mul_(255.0)
            loss = -model(x).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            losses.update(loss.item(), x.size(0))
            if i % args.print_freq == 0:
                 progress.display(i)

            if i % args.save_freq == 0:
                save(epoch, global_step)
                generate(epoch, global_step)
                model.train()
            global_step += 1
        return global_step

    for epoch in range(start_epoch, start_epoch + args.epoch):
        global_step = train(epoch, global_step)

# https://github.com/pytorch/examples/blob/master/imagenet/main.py#L363

class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Glow')

    parser.add_argument('--data-dir', type=str, metavar='DIR', required=True)
    parser.add_argument('--batch-size', type=int, default=32, metavar='N')
    parser.add_argument('--num-workers', type=int, default=2, metavar='N')

    parser.add_argument('--seed', default=40, type=int, metavar='N')

    parser.add_argument('--resume', type=str, metavar='FILE')
    parser.add_argument('--model-dir', type=str, default='checkpoint', metavar='DIR')
    parser.add_argument('--images', type=str, default='images', metavar='DIR')
    parser.add_argument('--print-freq', type=int, default=1, metavar='N')
    parser.add_argument('--save-freq', type=int, default=100, metavar='N')
    parser.add_argument('--epoch', type=int, default=1000, metavar='N')

    parser.add_argument('--image-size', type=int, default=64, metavar='N')
    parser.add_argument('--in-channels', type=int, default=3, metavar='N')
    parser.add_argument('--n-levels', type=int, default=4, metavar='N')
    parser.add_argument('--depth', type=int, default=48, metavar='N')
    parser.add_argument('--hidden_channels', type=int, default=512, metavar='N')

    parser.add_argument('--lr', type=float, default=0.001, metavar='F')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='F')
    main(parser.parse_args())
