import os
import matplotlib.pyplot as plt


def resolve_path(relative_path, caller_file):
    """Build an absolute path relative to a caller script's directory.

    Args:
        relative_path: file name or relative path to resolve.
        caller_file: __file__ of the calling module.

    Returns:
        Absolute path string.

    Example:
        path = resolve_path("data.csv", __file__)
    """
    return os.path.join(os.path.dirname(os.path.abspath(caller_file)), relative_path)


def save_figure(fig, out_path, dpi=150):
    """Save a matplotlib figure to disk and close it.

    Args:
        fig: matplotlib Figure object.
        out_path: absolute path to save the PNG.
        dpi: image resolution (default: 150).
    """
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {out_path}")
