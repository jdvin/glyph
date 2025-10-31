from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

import mne
import numpy as np
from textual_plotext import PlotextPlot
from textual.widgets import Label, Static
import matplotlib.cm as cm
from wavelet.spatial_tools import project_3d_to_2d
from textual.app import ComposeResult

from glyph.utils import Montage


def rgb_to_ansi_index(rgb: np.ndarray) -> list[int]:
    """
    Map an RGB triplet in [0,1] to the nearest 256-color ANSI index.
    """
    # Clamp input
    rgb = np.clip(rgb, 0, 1).round().astype(int)
    # Scale to 0..5 cube coordinates
    cube_index = 16 + 36 * rgb[:, 0] + 6 * rgb[:, 1] + rgb[:, 2]
    return cube_index.tolist()


def norm_to_ansi_index(z: float | np.ndarray, cmap: str = "plasma") -> int | list[int]:
    """Map a normalized value to a color using a colormap."""
    if isinstance(z, float):
        z = np.array([z])
    rgb_ = cm.get_cmap(cmap)(z)
    assert isinstance(rgb_, np.ndarray)
    ansi = rgb_to_ansi_index(rgb_)
    return ansi[0] if len(ansi) == 1 else ansi


class ChannelLinePlot(Static):
    """Widget that renders a channel line graph with proper µV axis."""

    def __init__(self, name: str, max_samples: int, ylim_uv: Optional[float]) -> None:
        super().__init__(classes="channel-plot")
        self._name = name
        self._buffer = deque(maxlen=max_samples)
        self._ylim = float(ylim_uv) if ylim_uv is not None else None

        self._title = Label(name, classes="channel-title")
        self._plot = PlotextPlot(id="channelplot")
        self._latest = Label("—", classes="channel-latest")

    def compose(self) -> ComposeResult:
        yield self._title
        yield self._plot
        yield self._latest

    def extend(self, values: Iterable[float]) -> None:
        # Append & plot
        new_values = list(values)
        if not new_values:
            return
        self._buffer.extend(new_values)

        y = np.asarray(self._buffer, dtype=float)

        # Choose y-limits
        if self._ylim is not None:
            y_min, y_max = -self._ylim, self._ylim
        else:
            # Autoscaling by median and median absolute deviation.
            med = float(np.median(y)) if y.size else 0.0
            mad = float(np.median(np.abs(y - med))) if y.size else 0.0
            robust_sigma = 1.4826 * mad
            pad = max(10.0, 2.0 * robust_sigma)  # ensure at least ±10 µV visible
            y_min, y_max = med - pad, med + pad
            if y_min == y_max:
                y_min, y_max = med - 1.0, med + 1.0

        # Plot with plotext
        plt = self._plot.plt
        plt.clear_figure()
        plt.title(self._name)
        plt.ylabel("µV")
        plt.plot(y.tolist(), marker="braille")  # x = sample index
        plt.ylim(y_min, y_max)
        plt.xlim(max(0, len(y) - len(self._buffer)), len(y) - 1)

        # Show latest sample (µV)
        self._latest.update(f"{new_values[-1]: .2f} µV")

        # Ask Textual to redraw this widget
        self._plot.refresh(layout=True)


def idw_grid(
    samples_xy: np.ndarray,
    samples_val: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    power: float = 2.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Inverse-Distance Weighting (IDW) onto a grid (no SciPy).
    samples_xy: (N,2) in [-1,1]x[-1,1]
    samples_val: (N,)
    grid_x, grid_y: meshgrid arrays (H,W)
    """
    N = samples_xy.shape[0]
    H, W = grid_x.shape
    Z = np.zeros((H, W), float)

    # Flatten grid for vectorized distances
    gx = grid_x.ravel()
    gy = grid_y.ravel()
    Zf = np.zeros_like(gx)
    wsum = np.zeros_like(gx)
    for i in range(N):
        dx = gx - samples_xy[i, 0]
        dy = gy - samples_xy[i, 1]
        d2 = dx * dx + dy * dy
        w = 1.0 / np.maximum(d2, eps) ** (power / 2.0)
        Zf += w * samples_val[i]
        if i == 0:
            wsum = w.copy()
        else:
            wsum += w
    Z = (Zf / np.maximum(wsum, eps)).reshape(H, W)
    return Z


# ──────────────────────────────────────────────────────────────────────────────
# The widget: Plotext underlay + electrode labels on top
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ElectrodeStyle:
    cmap: str = "plasma"  # plotext colormap name (fallback handled)
    radius: float = 0.98  # draw a head circle at this radius
    label_color: str = "white"
    label_shift: tuple[float, float] = (0.0, 0.0)  # nudge labels (dx, dy)
    show_head_circle: bool = True


class GlyphicMap(Static):
    """
    A Textual widget that draws a 2-D electrode map with a plotext underlay
    (heatmap via IDW or scatter), and labels per electrode.

    Positions must be in 2-D (range roughly [-1, 1]). Use `project_3d_to_2d`
    beforehand if you start from 3-D coordinates.

    Methods:
      - update_values({name: value})  # feed values for the underlay
      - set_positions({name: (x, y)}) # replace/update coordinates
      - set_labels_visible(bool)
    """

    def __init__(
        self,
        montage: Montage,
        grid_size: tuple[int, int] = (40, 20),  # (W, H) for heatmap
        style: ElectrodeStyle = ElectrodeStyle(),
        title: Optional[str] = "Electrode Map",
    ):
        super().__init__()
        self._plot = PlotextPlot(id="map-plot")
        self._montage = montage  # {'Fp1':(-0.8,0.9), ...}
        self._ch_names = [ch.reference_label for ch in montage.channel_map]
        self._ch_labels = [
            f"{ch.board_label}|{ch.reference_label}" for ch in montage.channel_map
        ]
        ch_positions = mne.channels.make_standard_montage(
            montage.reference_system
        ).get_positions()["ch_pos"]
        pos3d = np.vstack([ch_positions[name] for name in self._ch_names])
        self._positions = project_3d_to_2d(
            pos3d,
            sphere=(0.0, 0.01, 0.0, 0.0),
        )
        self._values = np.zeros(len(self._positions))

        self._grid_w, self._grid_h = grid_size
        self._style = style
        self._title = title
        self._show_labels = True
        self.off = 0.01

    def compose(self):
        yield self._plot

    def update_values(self, values: np.ndarray):
        self._values = self._norm_z(values)
        self._redraw()

    # ── internals ────────────────────────────────────────────────────────────
    def _norm_xy(self, xy: np.ndarray) -> np.ndarray:
        """Ensure coords live in a gently padded [-1,1] box."""
        return np.clip(xy, -1.0, 1.0)

    def _norm_z(self, z: np.ndarray) -> np.ndarray:
        """Normalize to 0..1 for coloring; robust against outliers."""
        # Normalize to 0..1 for coloring; robust against outliers
        med = float(np.median(z))
        mad = float(np.median(np.abs(z - med))) + 1e-9
        Zn = 0.5 + 0.25 * (z - med) / (1.4826 * mad)  # clamp to ~[0,1]
        Zn = np.clip(Zn, 0.0, 1.0)
        return Zn

    def _build_underlay(self, plt):
        # Grid in [-1,1] × [-1,1] (Y first because imshow expects matrix rows as Y)
        gx = np.linspace(-1.0, 1.0, self._grid_w)
        gy = np.linspace(-1.0, 1.0, self._grid_h)
        Gx, Gy = np.meshgrid(gx, gy)

        # Inverse-distance interpolation for smooth underlay
        Zn = idw_grid(self._positions, self._values, Gx, Gy, power=3.0)
        flatx, flaty, flatz = Gx.ravel(), Gy.ravel(), Zn.ravel()
        clrs = norm_to_ansi_index(flatz, self._style.cmap)
        plt.scatter(flatx.tolist(), flaty.tolist(), color=clrs, marker="■")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        # Optional head outline
        if self._style.show_head_circle:
            th = np.linspace(0, 2 * np.pi, 360)
            rx = self._style.radius * np.sin(th)
            ry = self._style.radius * np.cos(th)
            plt.plot(rx.tolist(), ry.tolist())

    def _overlay_labels(self, plt):
        dx, dy = self._style.label_shift
        for label, xy, value in zip(self._ch_labels, self._positions, self._values):
            x, y = self._norm_xy(xy)
            color = norm_to_ansi_index(value, self._style.cmap)
            if self._show_labels:
                plt.text(
                    label,
                    float(x + dx),
                    float(y + dy),
                    color=color,
                )
            else:
                # small dot when labels off
                plt.scatter([x], [y], marker="·", color=self._style.label_color)

    def on_show(self) -> None:
        # initial draw
        self._redraw()

    def _redraw(self):
        plt = self._plot.plt
        plt.clear_figure()
        if self._title:
            plt.title(self._title)

        self._build_underlay(plt)
        self._overlay_labels(plt)

        # keep coordinate frame consistent
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        self._plot.refresh(layout=True)
