"""Animated class -> expert usage heatmap and overall expert utilisation over training.

Re-render from a finished run:  python -m moe.animate runs/assignment
"""
import argparse
import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from .data import CLASSES
from .losses import class_to_expert

HISTORY_FILE = 'usage_history.npz'


def build_figure(history, assigned=None):
    """Returns (figure, update) where update(i, caption) draws epoch i of `history` (epochs, classes, experts)."""
    experts = history.shape[2]
    fig, (heat, bar) = plt.subplots(1, 2, figsize=(7.5, 3.3), dpi=70, gridspec_kw={'width_ratios': [2, 1]})
    image = heat.imshow(history[0], cmap='YlOrRd', vmin=0, vmax=history.max(), aspect='auto')
    fig.colorbar(image, ax=heat, label='Mean gate weight')
    heat.set_xticks(range(experts), [f'E{i}' for i in range(experts)], fontsize=7)
    heat.set_yticks(range(len(CLASSES)), CLASSES, fontsize=7)
    heat.set_title('Expert usage by class', fontsize=9)
    for cls, expert in enumerate(assigned if assigned is not None else []):
        heat.add_patch(Rectangle((expert - 0.5, cls - 0.5), 1, 1, fill=False, edgecolor='k', linewidth=0.9))
    bars = bar.bar(range(experts), history[0].mean(0))
    bar.axhline(1 / experts, color='gray', linestyle='--', linewidth=0.8)
    bar.set_ylim(0, history.mean(1).max() * 1.15)
    bar.set_xticks(range(experts))
    bar.tick_params(labelsize=7)
    bar.set(xlabel='Expert ID', title='Overall utilisation')
    bar.title.set_fontsize(9)
    caption = fig.suptitle('', fontsize=9)
    fig.tight_layout()

    def update(i, text):
        image.set_data(history[i])
        for rect, height in zip(bars, history[i].mean(0)):
            rect.set_height(height)
        caption.set_text(text)

    return fig, update


def save_animation(history, captions, out_dir, assigned=None, fps=3, hold=6):
    """Writes usage.gif (plus usage.mp4 if ffmpeg exists) and usage_final.png into `out_dir`."""
    fig, update = build_figure(history, assigned)
    last = len(history) - 1
    frames = list(range(len(history))) + [last] * hold
    anim = FuncAnimation(fig, lambda i: update(i, captions[i]), frames=frames)
    anim.save(os.path.join(out_dir, 'usage.gif'), writer=PillowWriter(fps=fps))
    if shutil.which('ffmpeg'):
        anim.save(os.path.join(out_dir, 'usage.mp4'), writer=FFMpegWriter(fps=fps, bitrate=300))
    update(last, captions[last])
    fig.savefig(os.path.join(out_dir, 'usage_final.png'))
    plt.close(fig)


def make_captions(variant, train_acc, val_acc):
    return [f'{variant} | epoch {i + 1} | train {tr:.1f}% | val {va:.1f}%'
            for i, (tr, va) in enumerate(zip(train_acc, val_acc))]


def save_history(out_dir, variant, usage, train_acc, val_acc):
    np.savez(os.path.join(out_dir, HISTORY_FILE), variant=variant, usage=np.stack(usage),
             train_acc=train_acc, val_acc=val_acc)


def render_run(out_dir, assignment):
    data = np.load(os.path.join(out_dir, HISTORY_FILE))
    usage = data['usage']
    assigned = class_to_expert(len(CLASSES), usage.shape[2]).numpy() if assignment else None
    captions = make_captions(str(data['variant']), data['train_acc'], data['val_acc'])
    save_animation(usage, captions, out_dir, assigned)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--assignment', action='store_true', help='outline each class\'s assigned expert')
    args = parser.parse_args()
    render_run(args.run_dir, args.assignment)
