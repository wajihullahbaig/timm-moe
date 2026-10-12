import os

import numpy as np

from moe.animate import HISTORY_FILE, render_run, save_history


def test_render_run_writes_a_small_gif_and_final_png(tmp_path):
    rng = np.random.default_rng(0)
    usage = rng.random((4, 10, 5))
    usage /= usage.sum(2, keepdims=True)
    save_history(str(tmp_path), 'assignment', list(usage), [10, 20, 30, 40], [15, 25, 35, 45])
    assert os.path.exists(tmp_path / HISTORY_FILE)
    render_run(str(tmp_path), assignment=True)
    assert 0 < os.path.getsize(tmp_path / 'usage.gif') < 1_000_000
    assert os.path.exists(tmp_path / 'usage_final.png')
