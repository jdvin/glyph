"""Interactive TUI for streaming OpenBCI Cyton-Daisy data with BrainFlow (line plots, proper µV axes)."""

from __future__ import annotations

import argparse
import sys
from queue import Empty
from typing import List, Optional

import numpy as np
from brainflow.board_shim import (
    BoardShim,
    BrainFlowError,
)
from loguru import logger
from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.timer import Timer
from textual.widgets import Footer, Header, TabPane, TabbedContent

from .plot import ChannelLinePlot
from .streamer import BrainFlowStreamer, MockEEGStreamer, StreamerProtocol
from .utils import AppConfig, create_board, detect_serial_port, load_app_config


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
        "--mock-eeg",
        action="store_true",
        help="Use simulated EEG data instead of connecting to a BrainFlow-compatible board.",
    )
    return parser.parse_args()


class OpenBCIApp(App):
    """Textual application that renders live EEG line graphs."""

    CSS = """
    Screen { layout: vertical; }
    Header { dock: top; }
    Footer { dock: bottom; }

    #plots {
        layout: grid;
        grid-size: 4;
        grid-gutter: 0;
        padding: 0;
    }
    .channel-plot {
        border: round $surface;
        padding: 0;
        margin: 0;
    }
    .channel-title { text-style: bold; }
    .channel-latest { color: $accent; }

    /* Make each plot reasonably tall for detail */
    PlotextPlot {
        width: 100%;
        height: 14;
        padding: 0;
        margin: 0;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(
        self,
        streamer: StreamerProtocol,
        channel_indices: List[int],
        channel_names: List[str],
        window_size: int,
        refresh_interval: float,
        ylim_uv: Optional[float],
    ) -> None:
        super().__init__()
        self._streamer = streamer
        self._channel_indices = channel_indices
        self._channel_names = channel_names
        self._refresh_interval = refresh_interval
        self._ylim_uv = ylim_uv
        self._plots = [
            ChannelLinePlot(name, window_size, ylim_uv=self._ylim_uv)
            for name in self._channel_names
        ]
        self._refresh_timer: Optional[Timer] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent():
            with TabPane("Time Series"):
                with Grid(id="plots"):
                    for plot in self._plots:
                        yield plot
            with TabPane("Channel Map"):
                yield ...
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

    try:
        config: AppConfig = load_app_config()
    except ValueError as err:
        logger.error("Invalid configuration: {}", err)
        return 1

    board: Optional[BoardShim] = None
    streamer: Optional[StreamerProtocol] = None

    try:
        if args.mock_eeg:
            num_channels = 16
            channel_indices = list(range(num_channels))
            channel_names = [f"Mock-EEG-{i + 1}" for i in range(num_channels)]
            streamer = MockEEGStreamer(
                num_channels=num_channels,
                buffer_size=config.buffer_size,
                poll_interval=config.poll_interval,
            )
        else:
            BoardShim.enable_dev_board_logger()
            serial_port = args.serial_port or detect_serial_port()
            if serial_port is None:
                return 2

            board = create_board(serial_port)
            board_id = board.get_board_id()
            channel_indices = BoardShim.get_eeg_channels(board_id)
            channel_names = [f"EEG-{i + 1}" for i in range(len(channel_indices))]

            streamer = BrainFlowStreamer(board, config.buffer_size, config.poll_interval)

        app = OpenBCIApp(
            streamer=streamer,
            channel_indices=channel_indices,
            channel_names=channel_names,
            window_size=config.window_size,
            refresh_interval=config.refresh_interval,
            ylim_uv=config.ylim,
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
