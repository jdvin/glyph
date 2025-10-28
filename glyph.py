"""Interactive TUI for streaming OpenBCI Cyton-Daisy data with BrainFlow (line plots, proper µV axes)."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from queue import Empty, Queue
from typing import Iterable, List, Optional

import numpy as np
from brainflow.board_shim import (
    BoardIds,
    BoardShim,
    BrainFlowError,
    BrainFlowInputParams,
)
from loguru import logger
from serial.tools import list_ports
from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.timer import Timer
from textual.widgets import Footer, Header, Label, Static

# New: line plotting widget
from textual_plotext import PlotextPlot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream data from an OpenBCI Cyton-Daisy headset (Textual line plots)."
    )
    parser.add_argument(
        "--serial-port",
        default=None,
        help=(
            "Serial device for the Cyton-Daisy board "
            "(e.g., /dev/tty.usbserial-XXXX on macOS or COM3 on Windows)."
        ),
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=256,
        help="Number of samples per channel fetched on each read (default: 256).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.05,
        help="Seconds to wait between board reads when no data is returned (default: 0.05).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=512,
        help="Number of recent samples to display per channel (default: 512).",
    )
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=0.1,
        help="Seconds between UI refreshes (default: 0.1).",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        default=None,
        help="If set, fix y-limits to ±this many µV (e.g., 100). Otherwise uses robust auto-scale.",
    )
    parser.add_argument(
        "--detrend",
        action="store_true",
        help="Subtract running median per channel window to zero-center the plotted signal.",
    )
    return parser.parse_args()


def create_board(port: str) -> BoardShim:
    params = BrainFlowInputParams()
    params.serial_port = port
    return BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)


def _filter_candidate_ports(ports: Iterable) -> list:
    candidates = []
    for port in ports:
        description_bits = [
            getattr(port, "manufacturer", None),
            getattr(port, "description", None),
            getattr(port, "hwid", None),
        ]
        combined = " ".join(bit for bit in description_bits if bit)
        if any(
            keyword in combined.lower()
            for keyword in ("openbci", "ftdi", "usbserial", "ttyusb", "ttyacm")
        ):
            candidates.append(port)
    return candidates


def detect_serial_port() -> Optional[str]:
    ports = list(list_ports.comports())
    if not ports:
        logger.error(
            "No serial devices detected. Connect the OpenBCI dongle and try again."
        )
        return None

    candidates = _filter_candidate_ports(ports)
    if not candidates:
        logger.warning(
            "Serial devices detected but none matched typical OpenBCI identifiers. "
            "Falling back to the first available device ({})",
            ports[0].device,
        )
        return ports[0].device

    if len(candidates) == 1:
        device = candidates[0].device
        logger.info("Auto-detected OpenBCI board on {}", device)
        return device

    devices = ", ".join(port.device for port in candidates)
    logger.warning(
        "Multiple OpenBCI-like devices detected: {}. Using {}. "
        "Override with --serial-port if this is incorrect.",
        devices,
        candidates[0].device,
    )
    return candidates[0].device


class BrainFlowStreamer:
    """Background thread that feeds BrainFlow data into a queue for the UI."""

    def __init__(
        self, board: BoardShim, buffer_size: int, poll_interval: float
    ) -> None:
        self._board = board
        self._buffer_size = buffer_size
        self._poll_interval = poll_interval
        self.queue: Queue[np.ndarray] = Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stream_started = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._board.prepare_session()
        self._board.start_stream()
        self._stream_started = True

        def _run() -> None:
            logger.info("BrainFlow streaming thread started.")
            while not self._stop_event.is_set():
                try:
                    chunk = self._board.get_board_data(self._buffer_size)
                    if chunk.size > 0:
                        self.queue.put(chunk)
                    else:
                        time.sleep(self._poll_interval)
                except BrainFlowError as err:
                    logger.error("BrainFlow read error: {}", err)
                    break
            logger.info("BrainFlow streaming thread stopped.")

        self._thread = threading.Thread(
            target=_run, name="BrainFlowStreamer", daemon=True
        )
        self._thread.start()

    def request_stop(self) -> None:
        self._stop_event.set()

    def close(self) -> None:
        self.request_stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._stream_started:
            try:
                self._board.stop_stream()
            except BrainFlowError as err:
                logger.warning("Failed to stop BrainFlow stream cleanly: {}", err)
            try:
                self._board.release_session()
            except BrainFlowError as err:
                logger.warning("Failed to release BrainFlow session: {}", err)


class ChannelLinePlot(Static):
    """Widget that renders a channel line graph with proper µV axis."""

    def __init__(
        self, name: str, max_samples: int, ylim_uv: Optional[float], detrend: bool
    ) -> None:
        super().__init__(classes="channel-plot")
        self._name = name
        self._buffer = deque(maxlen=max_samples)
        self._ylim = float(ylim_uv) if ylim_uv is not None else None
        self._detrend = detrend

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

        # Optional display-time zero-centering for readability
        if self._detrend and y.size:
            med = np.median(y)
            y = y - med

        # Choose y-limits
        if self._ylim is not None:
            y_min, y_max = -self._ylim, self._ylim
        else:
            # robust: median ± 8 * MAD (≈ ~±6.0σ for Gaussian when using 1.4826 * MAD)
            med = float(np.median(y)) if y.size else 0.0
            mad = float(np.median(np.abs(y - med))) if y.size else 0.0
            robust_sigma = 1.4826 * mad
            pad = max(10.0, 8.0 * robust_sigma)  # ensure at least ±10 µV visible
            y_min, y_max = med - pad, med + pad
            if y_min == y_max:
                y_min, y_max = med - 1.0, med + 1.0

        # Plot with plotext
        plt = self._plot.plt
        plt.clear_figure()
        plt.title(self._name)
        plt.ylabel("µV")
        plt.plot(y.tolist())  # x = sample index
        plt.ylim(y_min, y_max)
        plt.xlim(max(0, len(y) - len(self._buffer)), len(y) - 1)

        # Show latest sample (µV)
        self._latest.update(f"{new_values[-1]: .2f} µV")

        # Ask Textual to redraw this widget
        self._plot.refresh(layout=True)


class OpenBCIApp(App):
    """Textual application that renders live EEG line graphs."""

    CSS = """
    Screen { layout: vertical; }
    Header { dock: top; }
    Footer { dock: bottom; }

    #plots {
        layout: grid;
        grid-size: 4;
        grid-gutter: 1 2;
        padding: 1;
    }
    .channel-plot {
        border: round $surface;
        padding: 1;
    }
    .channel-title { text-style: bold; }
    .channel-latest { color: $accent; }

    /* Make each plot reasonably tall for detail */
    PlotextPlot {
        width: 100%;
        height: 14;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(
        self,
        streamer: BrainFlowStreamer,
        channel_indices: List[int],
        channel_names: List[str],
        window_size: int,
        refresh_interval: float,
        ylim_uv: Optional[float],
        detrend: bool,
    ) -> None:
        super().__init__()
        self._streamer = streamer
        self._channel_indices = channel_indices
        self._channel_names = channel_names
        self._refresh_interval = refresh_interval
        self._ylim_uv = ylim_uv
        self._detrend = detrend
        self._plots = [
            ChannelLinePlot(
                name, window_size, ylim_uv=self._ylim_uv, detrend=self._detrend
            )
            for name in self._channel_names
        ]
        self._refresh_timer: Optional[Timer] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Grid(id="plots"):
            for plot in self._plots:
                yield plot
        yield Footer()

    async def on_mount(self) -> None:
        try:
            self._streamer.start()
        except BrainFlowError as err:
            logger.error("Failed to start BrainFlow stream: {}", err)
            self.exit(message=str(err))
            return
        self._refresh_timer = self.set_interval(
            self._refresh_interval, self._refresh_plots, pause=False
        )

    async def _refresh_plots(self) -> None:
        chunks = []
        while True:
            try:
                chunk = self._streamer.queue.get_nowait()
            except Empty:
                break
            chunks.append(chunk)

        if not chunks:
            return

        combined = np.concatenate(chunks, axis=1)

        # BrainFlow returns a [num_channels x num_samples] array.
        eeg_data = combined[self._channel_indices, :]

        # IMPORTANT: For OpenBCI via BrainFlow, EXG channels are already in µV.
        # Do NOT apply an additional scale factor here.

        for idx, row in enumerate(eeg_data):
            self._plots[idx].extend(row.tolist())

    async def on_shutdown(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        self._streamer.request_stop()

    def action_quit(self) -> None:
        self.exit()


def main() -> int:
    args = parse_args()
    BoardShim.enable_dev_board_logger()

    serial_port = args.serial_port or detect_serial_port()
    if serial_port is None:
        return 2

    board: Optional[BoardShim] = None
    streamer: Optional[BrainFlowStreamer] = None

    try:
        board = create_board(serial_port)
        board_id = board.get_board_id()
        channel_indices = BoardShim.get_eeg_channels(board_id)
        channel_names = [f"EEG {i + 1}" for i in range(len(channel_indices))]

        streamer = BrainFlowStreamer(board, args.buffer_size, args.poll_interval)
        app = OpenBCIApp(
            streamer=streamer,
            channel_indices=channel_indices,
            channel_names=channel_names,
            window_size=args.window_size,
            refresh_interval=args.refresh_interval,
            ylim_uv=args.ylim,
            detrend=args.detrend,
        )
        app.run()
    except KeyboardInterrupt:
        logger.info("Stream stopped by user.")
    except BrainFlowError as err:
        logger.error("BrainFlow error: {}", err)
        return 1
    finally:
        if streamer is not None:
            streamer.close()
        elif board is not None:
            try:
                board.release_session()
            except BrainFlowError:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
