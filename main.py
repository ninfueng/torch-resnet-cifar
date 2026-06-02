import argparse
import logging
import os
import time
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as T
from timm.utils.metrics import AverageMeter, accuracy
from torch.utils.data import DataLoader

import resnet


def train(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device,
) -> None:
    top1 = AverageMeter()
    losses = AverageMeter()

    model.train()
    start = time.perf_counter()

    for input, target in loader:
        input = input.to(device)
        target = target.to(device)

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        size = input.size(0)
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), size)
        top1.update(prec1.item(), size)

    runtime = time.perf_counter() - start
    logging.info(
        f'Tr E={epoch:03d}, A={top1.avg:.2f}, L={losses.avg:.4f}, {runtime:.3f}s'
    )


@torch.no_grad()
def test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.modules.loss._Loss,
    epoch: int,
    device,
) -> float:
    top1 = AverageMeter()
    losses = AverageMeter()

    model.eval()
    start = time.perf_counter()

    for input, target in loader:
        input = input.to(device)
        target = target.to(device)

        output = model(input)
        loss = criterion(output, target)

        size = input.size(0)
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), size)
        top1.update(prec1.item(), size)

    runtime = time.perf_counter() - start
    logging.info(
        f'Te E={epoch:03d}, A={top1.avg:.2f}, L={losses.avg:.4f}, {runtime:.3f}s'
    )
    return top1.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet20')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--exp-dir', type=str, default='exps')
    parser.add_argument('--dataset-dir', type=str, default='./data')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()

    cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    datetime_ = str(datetime.now()).replace(' ', '-')
    exp_dir = os.path.join(args.exp_dir, datetime_)
    os.makedirs(exp_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s %(asctime)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(exp_dir, 'info.log')),
            logging.StreamHandler(),
        ],
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet.__dict__[args.arch]()
    model = model.to(device)
    if args.compile:
        model = torch.compile(model)

    train_transforms = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, 4),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    val_transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset_dir = os.path.expanduser(args.dataset_dir)
    train_dataset = datasets.CIFAR10(
        root=dataset_dir,
        train=True,
        transform=train_transforms,
        download=True,
    )
    val_dataset = datasets.CIFAR10(
        root=dataset_dir, train=False, transform=val_transforms, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(args.batch_size * 1.5),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
    )

    best_prec1 = best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch, device)
        prec1 = test(model, val_loader, criterion, epoch, device)
        lr_scheduler.step()

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        best_epoch = epoch + 1
        if is_best:
            torch.save(
                {
                    'epoch': best_epoch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                },
                os.path.join(exp_dir, f'{args.arch}.pt'),
            )
    logging.info(f'Best E:{best_epoch}, A: {best_prec1:.2f}')
