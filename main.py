import argparse
import logging
import os
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.optim.lr_scheduler import MultiStepLR

import resnet
from train_utils import test, train

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
    train_dataset = CIFAR10(
        root=dataset_dir,
        train=True,
        transform=train_transforms,
        download=True,
    )
    test_dataset = CIFAR10(
        root=dataset_dir, train=False, transform=val_transforms, download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=round(args.batch_size * 1.5),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = MultiStepLR(
        optimizer,
        milestones=[100, 150],
    )

    best_prec1 = best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch, device)
        prec1 = test(model, test_loader, criterion, epoch, device)
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
