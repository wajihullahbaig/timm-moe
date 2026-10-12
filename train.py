"""Train a MoE variant on CIFAR-10:  python train.py --variant assignment --epochs 30

Model selection and early stopping use a held-out validation split; the test set is only reported.
"""
import argparse
import csv
import os
import time

import torch

from moe.animate import render_run, save_history
from moe.config import VARIANTS
from moe.data import cifar10_loaders
from moe.engine import EarlyStopping, build_optimizer, evaluate, set_seed, train_epoch
from moe.model import MoEClassifier


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, default='assignment')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=8, help='stop after this many epochs without val improvement')
    parser.add_argument('--backbone', default='resnet18', help='any timm model supporting features_only')
    parser.add_argument('--out-index', type=int, default=2, help='backbone feature stage fed to the MoE')
    parser.add_argument('--input-scale', type=int, default=2, help='integer upsampling of the 32px images before the backbone')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--out-dir', default=None, help='default: runs/<variant>')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-amp', action='store_true', help='disable mixed precision')
    parser.add_argument('--no-pretrained', action='store_true', help='random backbone (no weight download)')
    return parser.parse_args()


def log_row(path, row):
    new_file = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def build_model(args, variant, device):
    return MoEClassifier(variant.num_experts, temperature=variant.temperature, backbone=args.backbone,
                         out_index=args.out_index, input_scale=args.input_scale,
                         pretrained=not args.no_pretrained).to(device)


def main():
    args = parse_args()
    variant = VARIANTS[args.variant]
    out_dir = args.out_dir or os.path.join('runs', args.variant)
    checkpoint = os.path.join(out_dir, 'best_model.pth')
    os.makedirs(out_dir, exist_ok=True)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda' and not args.no_amp

    train_loader, val_loader, test_loader = cifar10_loaders(args.data_dir, args.batch_size, device)
    model = build_model(args, variant, device)
    optimizer, scheduler = build_optimizer(model, args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    stopper = EarlyStopping(args.patience)

    usage_history, train_accs, val_accs = [], [], []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        losses, train_acc = train_epoch(model, train_loader, optimizer, scheduler, scaler, variant.loss_weights, amp)
        val_acc, usage = evaluate(model, val_loader, amp)
        test_acc, _ = evaluate(model, test_loader, amp)
        seconds = time.time() - start
        print(f'epoch {epoch:3d} | train {train_acc:6.2f}% | val {val_acc:6.2f}% | test {test_acc:6.2f}% | '
              f'{seconds:5.1f}s | ' + ' '.join(f'{k}={v:.3f}' for k, v in losses.items()), flush=True)
        log_row(os.path.join(out_dir, 'metrics.csv'), {'epoch': epoch, 'train_acc': train_acc, 'val_acc': val_acc,
                                                       'test_acc': test_acc, 'seconds': seconds, **losses})
        usage_history.append(usage)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        if stopper.update(val_acc):
            torch.save(model.state_dict(), checkpoint)
        if stopper.should_stop:
            print(f'early stopping: no val improvement for {args.patience} epochs')
            break

    save_history(out_dir, args.variant, usage_history, train_accs, val_accs)
    render_run(out_dir, assignment='routing' in variant.loss_weights)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    final_test_acc, _ = evaluate(model, test_loader, amp)
    print(f'best val acc {stopper.best:.2f}% -> test acc of that checkpoint {final_test_acc:.2f}%')


if __name__ == '__main__':
    main()
