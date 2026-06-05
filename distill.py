"""Try to train FitNet."""

import argparse
import logging
import os
import time
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.utils.metrics import AverageMeter, accuracy
from torch import Tensor, nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import resnet
from forward_hook import ForwardHookManager
from train_utils import test


def set_forward_manager(
    teacher_model: nn.Module,
    student_model: nn.Module,
    teacher_student_layer_map: dict[str, list[str | nn.Module | None]],
    device,
) -> dict[str, ForwardHookManager]:

    teacher_forward_manager = ForwardHookManager(device)
    student_forward_manager = ForwardHookManager(device)

    for teacher_layer in teacher_student_layer_map.keys():
        if teacher_layer == 'base':
            continue

        student_layer, _, _, _ = teacher_student_layer_map[teacher_layer]
        teacher_forward_manager.add_hook(
            teacher_model, teacher_layer, requires_input=False, requires_output=True
        )
        student_forward_manager.add_hook(
            student_model, student_layer, requires_input=False, requires_output=True
        )

    return {
        'teacher_forward_manager': teacher_forward_manager,
        'student_forward_manager': student_forward_manager,
    }


def get_distill_losses(
    teacher_forward_manager: ForwardHookManager,
    student_forward_manager: ForwardHookManager,
    teacher_student_layer_map: dict[str, list[str | nn.Module]],
) -> list[Tensor]:

    teacher_io_dict = teacher_forward_manager.pop_io_dict()
    student_io_dict = student_forward_manager.pop_io_dict()

    distill_losses = []

    for teacher_layer in teacher_student_layer_map.keys():
        if teacher_layer == 'base':
            continue

        student_layer, feature_extractor, loss, weight = teacher_student_layer_map[
            teacher_layer
        ]

        teacher_out = teacher_io_dict[teacher_layer]['output']
        student_out = student_io_dict[student_layer]['output']

        student_out = feature_extractor(student_out)
        distill_loss = weight * loss(student_out, teacher_out)
        distill_losses.append(distill_loss)

    return distill_losses


def parse_auxiliary_module(
    teacher_student_layer_map: dict,
) -> dict[str, nn.Module]:

    auxiliary_map = {}

    for teacher_layer in teacher_student_layer_map.keys():
        if teacher_layer == 'base':
            continue

        student_layer, feature_extractor, _, _ = teacher_student_layer_map[
            teacher_layer
        ]
        auxiliary_layer = teacher_layer + '_' + student_layer
        # NOTE: . cannot be used with name of modules.
        auxiliary_layer = auxiliary_layer.replace('.', '_')
        auxiliary_map[auxiliary_layer] = feature_extractor

    return auxiliary_map


def train_distill(
    teacher_model: nn.Module,
    student_model: nn.Module,
    loader: DataLoader,
    criterion: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    teacher_forward_manager: ForwardHookManager,
    student_forward_manager: ForwardHookManager,
    teacher_student_layer_map: dict,
    device,
) -> None:

    teacher_model.eval()
    student_model.train()

    teacher_top1, teacher_losses = AverageMeter(), AverageMeter()
    student_top1, student_losses = AverageMeter(), AverageMeter()
    cross_entropy_weight = teacher_student_layer_map['base'][-1]

    start = time.perf_counter()

    for input, target in loader:
        input, target = input.to(device), target.to(device)

        student_output = student_model(input)

        with torch.no_grad():
            teacher_output = teacher_model(input)
            teacher_loss = criterion(teacher_output, target)

        student_loss = criterion(student_output, target)
        student_loss *= cross_entropy_weight

        distill_losses = get_distill_losses(
            teacher_forward_manager, student_forward_manager, teacher_student_layer_map
        )
        # NOTE: distill weights included in get_distill_losses().
        all_distill_loss = 0.0
        for distill_loss in distill_losses:
            all_distill_loss += distill_loss
        student_loss += all_distill_loss

        optimizer.zero_grad()
        student_loss.backward()
        optimizer.step()

        with torch.no_grad():
            size = input.size(0)

            teacher_prec1 = accuracy(teacher_output.data, target)[0]
            student_prec1 = accuracy(student_output.data, target)[0]

            student_losses.update(student_loss.item(), size)
            student_top1.update(student_prec1.item(), size)

            teacher_losses.update(teacher_loss.item(), size)
            teacher_top1.update(teacher_prec1.item(), size)

    runtime = time.perf_counter() - start

    logging.info(
        f'Tr E={epoch:03d}, '
        f'A={student_top1.avg:.2f}, '
        f'L={student_losses.avg:.4f}, '
        f'Lr={lr_scheduler.get_last_lr()[0]:.3e} '
        f'{runtime:.3f}s'
    )


def logit_distillation(
    teacher_logits: Tensor,
    student_logits: Tensor,
    T: float,
) -> Tensor:

    soft_teacher_logits = F.softmax(teacher_logits / T, dim=-1)
    soft_student_logits = F.log_softmax(student_logits / T, dim=-1)

    # NOTE: cross entropy between teacher to teacher - student
    distill_loss = (
        (T**2.0)
        * soft_teacher_logits
        * (soft_teacher_logits.log() - soft_student_logits)
    )
    distill_loss = distill_loss.mean()
    return distill_loss


class LogitDistillationLoss(nn.Module):
    def __init__(self, T: float) -> None:
        super().__init__()
        self.T = T

    def forward(self, teacher_logits: Tensor, student_logits: Tensor) -> Tensor:
        distill_loss = logit_distillation(teacher_logits, student_logits, self.T)
        return distill_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--exp-dir', type=str, default='exps')
    parser.add_argument('--dataset-dir', type=str, default='./data')

    parser.add_argument('--student-arch', type=str, default='resnet20')
    parser.add_argument('--teacher-arch', type=str, default='resnet110')
    parser.add_argument('--student-dir', type=str, default='')
    parser.add_argument('--teacher-dir', type=str, default='resnet110.pt')
    parser.add_argument('--T', type=float, default=2.0)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()

    # NOTE: teacher_layer: [student_layer, feature_extractor, loss, loss_weight]
    TEACHER_STUDENT_LAYER_MAP = {
        # cross entropy loss
        'base': [
            None,
            None,
            None,
            3 / 4,
        ],
        # # feature-based
        # 'layer3.17': [
        #     'layer3.0',
        #     nn.Identity(),
        #     nn.MSELoss(reduction='mean'),
        #     1 / 8,
        # ],
        # logit-based
        'linear': [
            'linear',
            nn.Identity(),
            LogitDistillationLoss(T=args.T),
            1 / 4,
        ],
    }
    auxiliary_map = parse_auxiliary_module(TEACHER_STUDENT_LAYER_MAP)

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

    teacher_dir = os.path.expanduser(args.teacher_dir)
    student_dir = os.path.expanduser(args.student_dir)

    teacher_state_dict = None
    if os.path.isfile(teacher_dir):
        teacher_state_dict = torch.load(teacher_dir, weights_only=False)['state_dict']
    else:
        raise NotImplementedError('')

    student_state_dict = None
    if os.path.isfile(student_dir):
        student_state_dict = torch.load(student_dir, weights_only=False)['state_dict']
    else:
        logging.warning(f'student_dir is not found. Your: {student_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model = resnet.__dict__[args.teacher_arch]()
    student_model = resnet.__dict__[args.student_arch]()

    if teacher_state_dict is not None:
        teacher_model.load_state_dict(teacher_state_dict)

    if student_state_dict is not None:
        student_model.load_state_dict(student_state_dict)

    for name, module in auxiliary_map.items():
        student_model.add_module(name, module)

    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    if args.compile:
        teacher_model = torch.compile(teacher_model)
        student_model = torch.compile(student_model)

    train_transforms = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, 4),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    test_transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
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
        root=dataset_dir, train=False, transform=test_transforms, download=True
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
        student_model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = MultiStepLR(
        optimizer,
        milestones=[100, 150],
    )

    # NOTE: requires two forward managers, otherwise same layer names will cause
    # overlapped.
    forward_managers = set_forward_manager(
        teacher_model,
        student_model,
        TEACHER_STUDENT_LAYER_MAP,
        device,
    )
    teacher_forward_manager = forward_managers['teacher_forward_manager']
    student_forward_manager = forward_managers['student_forward_manager']

    best_student_prec1 = best_student_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_distill(
            teacher_model,
            student_model,
            train_loader,
            criterion,
            optimizer,
            lr_scheduler,
            epoch,
            teacher_forward_manager,
            student_forward_manager,
            TEACHER_STUDENT_LAYER_MAP,
            device,
        )
        student_prec1 = test(
            student_model,
            test_loader,
            criterion,
            lr_scheduler,
            epoch,
            device,
        )
        lr_scheduler.step()

        state_dict = student_model.state_dict()
        if hasattr(student_model, '_orig_mod'):
            state_dict = student_model._orig_mod.state_dict()

        is_best = student_prec1 > best_student_prec1
        best_student_prec1 = max(student_prec1, best_student_prec1)
        best_student_epoch = epoch + 1
        if is_best:
            torch.save(
                {
                    'epoch': best_student_epoch,
                    'state_dict': state_dict,
                    'best_student_prec1': best_student_prec1,
                },
                os.path.join(exp_dir, f'{args.student_arch}.pt'),
            )
    logging.info(f'Best E:{best_student_epoch}, A: {best_student_prec1:.2f}')
