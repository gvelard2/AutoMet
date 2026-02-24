import os
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from automet.utils import resolve_path, save_figure


def test_resolve_path_returns_absolute():
    result = resolve_path("data.csv", __file__)
    assert os.path.isabs(result)


def test_resolve_path_same_dir():
    result = resolve_path("data.csv", __file__)
    expected = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.csv")
    assert result == expected


def test_resolve_path_subdirectory():
    result = resolve_path(os.path.join("sub", "file.txt"), __file__)
    assert result.endswith(os.path.join("sub", "file.txt"))


def test_save_figure_creates_file(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    out_path = str(tmp_path / "test_output.png")
    save_figure(fig, out_path, dpi=72)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_save_figure_closes_figure(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out_path = str(tmp_path / "closed.png")
    open_before = plt.get_fignums()
    save_figure(fig, out_path, dpi=72)
    open_after = plt.get_fignums()
    assert fig.number not in open_after
