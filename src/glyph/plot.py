from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

import mne
import numpy as np
from textual_plotext import PlotextPlot
from textual.widgets import Label, Static

# from wavelet.spatial_tools import project_3d_to_2d
from textual.app import ComposeResult

from glyph.utils import Montage


class ChannelLinePlot(Static):
    """Widget that renders a channel line graph with proper µV axis."""

    def __init__(self, name: str, max_samples: int, ylim_uv: Optional[float]) -> None:
        super().__init__(classes="channel-plot")
        self._name = name
        self._buffer = deque(maxlen=max_samples)
        self._ylim = float(ylim_uv) if ylim_uv is not None else None

        self._title = Label(name, classes="channel-title")
        self._plot = PlotextPlot(id=f"plot-{name}")
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
            pad = max(10.0, 4.0 * robust_sigma)  # ensure at least ±10 µV visible
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


def project_3d_to_2d(
    ch_pos_3d: np.ndarray,
    sphere: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.095),
    eeglab: bool = False,
) -> np.ndarray:
    """
    Convert 3D EEG electrode positions into 2D topomap coordinates.

    Parameters
    ----------
    ch_pos_3d :np.ndarray
        (x, y, z) in meters.
    sphere : tuple of (x, y, z, r)
        Center (x, y, z) and radius of the reference sphere.
        Use MNE's defaults: (0, 0, 0, 0.095) ≈ head radius 9.5 cm.
    eeglab : bool
        If True, use EEGLAB-style polar projection (slightly spreads rim electrodes).
        If False, use MNE's default orthographic / azimuthal projection.

    Returns
    -------
    np.ndarray
        (x2d, y2d)
    """
    sphere_center = np.array(sphere[:3], float)
    sphere_radius = float(sphere[3])
    coords_2d = np.zeros((len(ch_pos_3d), 2))
    # breakpoint()
    for i, xyz in enumerate(ch_pos_3d):
        # center and normalize to unit sphere
        xyz = np.asarray(xyz, float) - sphere_center
        r = np.linalg.norm(xyz)
        if r == 0:
            coords_2d[i] = np.zeros(2)
            continue
        xyz /= r

        x, y, z = xyz
        # spherical coordinates
        azimuth = np.arctan2(y, x)  # −π..π
        elevation = np.arcsin(z)  # −π/2..π/2

        if eeglab:
            # EEGLAB uses radius ∝ (π/2 − elevation)/(π/2)
            radius = (np.pi / 2 - elevation) / (np.pi / 2)
        else:
            # MNE default: radius ∝ cos(elevation)
            radius = np.cos(elevation)

        x2d = radius * np.cos(azimuth)
        y2d = radius * np.sin(azimuth)
        coords_2d[i] = np.array([x2d, y2d])

    return coords_2d


class ChannelMap(Static):
    """Simple ASCII topographic map with labels."""

    def __init__(self, montage: Montage):
        super().__init__()
        self.montage = montage  # {'Fp1':(-0.8,0.9), ...}
        self.ch_names = [ch.reference_label for ch in montage.channel_map]
        ch_positions = mne.channels.make_standard_montage(
            montage.reference_system
        ).get_positions()["ch_pos"]
        pos3d = np.vstack([ch_positions[name] for name in self.ch_names])
        y_diam = pos3d[:, 1].max() - pos3d[:, 1].min()
        self.positions = project_3d_to_2d(
            pos3d,
            sphere=(0.0, 0.01, 0.0, 0.0),
        )

        self.values = [0.0 for ch in self.positions]

    def update_values(self, values: list[float]):
        self.values = values
        self.refresh(layout=True)

    def render(self) -> str:
        h, w = 50, 150
        canvas = [[" "] * w for _ in range(h)]
        for ch, amp, (x, y) in zip(
            self.montage.channel_map, self.values, self.positions
        ):
            # normalize −1→1 to screen coordinates
            i = int((1 - y) * (h - 1) / 2)
            j = int((x + 1) * (w - 1) / 2)
            amp = abs(amp)
            mark = "." if amp < 20 else "o" if amp < 100 else "O"
            label = f"{mark} - {ch.board_label}|{ch.reference_label}"
            for k, c in enumerate(label):
                if 0 <= j + k < w:
                    canvas[i][j + k] = c
        return "\n".join("".join(row) for row in canvas)
