from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

import numpy as np
from textual_plotext import PlotextPlot
from textual.widgets import Label, Static

from textual.app import ComposeResult


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
