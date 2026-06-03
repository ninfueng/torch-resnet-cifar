import logging
import time

import torch
import torch.nn as nn
import torch.optim
from timm.utils.metrics import AverageMeter, accuracy
from torch.utils.data import DataLoader


def train(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device,
) -> None:
    top1, losses = AverageMeter(), AverageMeter()

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

        with torch.no_grad():
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
    top1, losses = AverageMeter(), AverageMeter()

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
