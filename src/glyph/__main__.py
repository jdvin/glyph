"""Interactive TUI for streaming OpenBCI Cyton-Daisy data with BrainFlow (line plots, proper µV axes)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from queue import Empty
from typing import List, Optional

import numpy as np
from brainflow.board_shim import (
    BoardShim,
    BrainFlowError,
    BoardIds,
)
from loguru import logger
from textual.app import App, ComposeResult
from textual.containers import CenterMiddle, Grid, Container
from textual.timer import Timer
from textual.widgets import Footer, Header, TabPane, TabbedContent, Static

from .plot import ChannelLinePlot, GlyphicMap
from .streamer import BrainFlowStreamer, MockEEGStreamer, StreamerProtocol
from .utils import AppConfig, Montage, create_board, detect_serial_port, load_app_config


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


@dataclass(frozen=True)
class BoardDetails:
    source: str
    name: str
    board_id: Optional[int]
    sampling_rate_hz: Optional[float]
    eeg_channel_count: Optional[int]
    serial_port: Optional[str] = None


class BoardDetailsPanel(Static):
    """Simple text panel summarizing board metadata."""

    def __init__(self, details: BoardDetails) -> None:
        super().__init__(id="board-details")
        self._details = details

    def on_mount(self) -> None:
        self.update(self._render_text())

    def update_details(self, details: BoardDetails) -> None:
        self._details = details
        self.update(self._render_text())

    def _render_text(self) -> str:
        details = self._details
        lines = ["Board Info", ""]
        lines.append(f"Source: {details.source}")
        lines.append(f"Name: {details.name}")
        if details.board_id is not None:
            lines.append(f"Board ID: {details.board_id}")
        if details.serial_port:
            lines.append(f"Serial Port: {details.serial_port}")
        if details.sampling_rate_hz is not None:
            lines.append(f"Sampling Rate: {details.sampling_rate_hz:.0f} Hz")
        if details.eeg_channel_count is not None:
            lines.append(f"EEG Channels: {details.eeg_channel_count}")
        return "\n".join(lines)


def _format_board_name(board_id: int) -> str:
    try:
        enum_name = BoardIds(board_id).name
    except ValueError:
        return f"Board {board_id}"
    return enum_name.replace("_", " ").title()


class Glyph(App):
    """Textual application that renders live EEG."""

    CSS = """
    Screen { layout: vertical; }
    Header { dock: top; }
    Footer { dock: bottom; }

    #channel-map {
        layout: grid;
        grid-size: 2;            /* two columns */
        grid-gutter: 1;
        padding: 1 2;
    }
    /* Allow the map widget to shrink within its grid cell */
    #mapplot { 
        min-width: 0;
    }

    #timeseries{
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
    .channel-latest { color: white; }

    #channelplot {
        width: 100%;
        height: 10;
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
        montage: Montage,
        ylim_uv: Optional[float],
        board_details: BoardDetails,
    ) -> None:
        super().__init__()
        self._streamer = streamer
        self._channel_indices = channel_indices
        self._channel_names = channel_names
        self._montage = montage
        self._refresh_interval = refresh_interval
        self._ylim_uv = ylim_uv
        self._board_details = board_details
        self._timeseries = [
            ChannelLinePlot(name, window_size, ylim_uv=self._ylim_uv)
            for name in self._channel_names
        ]
        self._channel_map = GlyphicMap(montage)
        self._board_panel = BoardDetailsPanel(board_details)
        self._refresh_timer: Optional[Timer] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent():
            with TabPane("Channel Map"):
                # with CenterMiddle(
                #     id="channel-map"
                # ):  # TODO: figure out why this is not working
                with Container(id="channel-map"):
                    yield self._channel_map
                    yield self._board_panel
            with TabPane("Time Series"):
                with Grid(id="timeseries"):
                    for plot in self._timeseries:
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

        for idx, row in enumerate(eeg_data):
            self._timeseries[idx].extend(row.tolist())
        import random

        self._channel_map.update_values(eeg_data[:, -1] * random.random())

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
    board_details: Optional[BoardDetails] = None

    try:
        montage = Montage.from_json(config.montage_path)
        channel_map_indices = [channel.index for channel in montage.channel_map]
        channel_names = [channel.reference_label for channel in montage.channel_map]
        if args.mock_eeg:
            num_channels = len(montage.channel_map)
            mock_sampling_rate = 250.0
            streamer = MockEEGStreamer(
                num_channels=num_channels,
                buffer_size=config.buffer_size,
                poll_interval=config.poll_interval,
                sampling_rate=mock_sampling_rate,
            )
            board_details = BoardDetails(
                source="Mock",
                name="Mock EEG Stream",
                board_id=None,
                sampling_rate_hz=mock_sampling_rate,
                eeg_channel_count=num_channels,
            )
        else:
            BoardShim.enable_dev_board_logger()
            serial_port = args.serial_port or detect_serial_port()
            if serial_port is None:
                return 2

            board = create_board(serial_port)
            board_id = board.get_board_id()
            sampling_rate = BoardShim.get_sampling_rate(board_id)
            board_channels = BoardShim.get_eeg_channels(board_id)
            assert (
                len(board_channels) == len(channel_map_indices)
            ), f"Board channel indices ({len(board_channels)}) do not match the configured channel indices ({len(channel_map_indices)})."
            streamer = BrainFlowStreamer(
                board, config.buffer_size, config.poll_interval
            )
            board_details = BoardDetails(
                source="BrainFlow",
                name=_format_board_name(board_id),
                board_id=board_id,
                sampling_rate_hz=float(sampling_rate),
                eeg_channel_count=len(board_channels),
                serial_port=serial_port,
            )
        assert streamer is not None
        assert board_details is not None
        app = Glyph(
            streamer=streamer,
            channel_indices=channel_map_indices,
            channel_names=channel_names,
            window_size=config.window_size,
            refresh_interval=config.refresh_interval,
            ylim_uv=config.ylim,
            montage=montage,
            board_details=board_details,
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
