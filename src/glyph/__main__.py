"""Interactive TUI for streaming OpenBCI Cyton-Daisy data with BrainFlow (line plots, proper µV axes)."""

from __future__ import annotations

import argparse
import sys
from queue import Empty
from typing import List, Optional
import json

import numpy as np
from brainflow.board_shim import (
    BoardShim,
    BrainFlowError,
)
from loguru import logger
from textual.app import App, ComposeResult
from textual.containers import Grid, Container
from textual.timer import Timer
from textual.widgets import Footer, Header, TabPane, TabbedContent, Select

from .plot import (
    ChannelLinePlot,
    GlyphicMap,
    BoardDetailsPanel,
    HealthMetricsPanel,
    BoardDetails,
    ModelDetailsPanel,
)
from .streamer import BrainFlowStreamer, MockEEGStreamer, StreamerProtocol
from .utils import (
    AppConfig,
    ModelLoaderConfig,
    Montage,
    create_board,
    detect_serial_port,
    load_app_config,
    format_board_name,
    load_model,
)


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
    parser.add_argument(
        "--model-loader-config-path",
        default=None,
        help="Path to a JSON file containing the model loader config.",
    )
    return parser.parse_args()


class Glyph(App):
    """Textual application that renders live EEG."""

    CSS = """
    Screen { layout: vertical; }
    Header { dock: top; }
    Footer { dock: bottom; }

    #channel-map {
        layout: grid;
        grid-size: 3;            /* add left metrics column */
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
        board_details: BoardDetails,
        model_loader_config_path: str | None,
    ) -> None:
        super().__init__()
        self._streamer = streamer
        self._channel_indices = channel_indices
        self._channel_names = channel_names
        self._montage = montage
        self._refresh_interval = refresh_interval
        self._ylim_uv = None
        self._board_details = board_details
        self._timeseries = [
            ChannelLinePlot(name, window_size, ylim_uv=self._ylim_uv)
            for name in self._channel_names
        ]
        self._channel_map = GlyphicMap(montage)
        self._health_panel = HealthMetricsPanel(self._channel_names)
        self._board_panel = BoardDetailsPanel(board_details)
        self._refresh_timer: Optional[Timer] = None
        if model_loader_config_path:
            logger.info("Loading model from {}", model_loader_config_path)
            with open(model_loader_config_path, "r", encoding="utf-8") as file:
                self._model_loader_config = ModelLoaderConfig(**json.load(file))
            self.model = load_model(self._model_loader_config)
        self._model_panel = ModelDetailsPanel(self.model)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent():
            with TabPane("Channel Map"):
                with Container(id="channel-map"):
                    yield self._board_panel
                    yield self._channel_map
                    yield self._health_panel
            with TabPane("Time Series"):
                # Y-limit selector above the plots
                yield Select(
                    options=[
                        ("200 µV", "200"),
                        ("100 µV", "100"),
                        ("50 µV", "50"),
                        ("Auto", "auto"),
                    ],
                    value=(
                        "auto" if self._ylim_uv is None else str(int(self._ylim_uv))
                    ),
                    id="ylim-select",
                )
                with Grid(id="timeseries"):
                    for plot in self._timeseries:
                        yield plot
            with TabPane("Model"):
                with Container(id="model-details"):
                    yield self._model_panel
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
        # BrainFlow returns a [num_channels + 3 x num_samples] array.
        # Index 0 is a sample index.
        # Indexes num_channels + [2-4] are accelerometer data.
        eeg_data = self._streamer.buffer.get_eeg()

        # IMPORTANT: For OpenBCI via BrainFlow, EXG channels are already in µV.

        for idx, row in enumerate(eeg_data):
            self._timeseries[idx].post(row)

        # Update health metrics panel; show waiting progress until ready.
        scores = np.zeros(len(self._channel_names), dtype=int)
        try:
            hm = self._streamer.health_metrics
            buf = self._streamer.buffer
            if buf.ready(hm.N):
                scores, flags = hm.compute(buf)
                self._health_panel.update_metrics(scores, flags)
            else:
                seconds_filled = buf.prop_filled(hm.N) * hm.window_sec
                self._health_panel.update_waiting(seconds_filled, hm.window_sec)
        except Exception as e:
            # Avoid crashing the UI if health computation fails
            logger.debug(f"Health metrics update skipped: {e}")
            raise e

        self._channel_map.update_values(scores)

        # channel_signals = torch.tensor(eeg_data)
        # channel_positions = self._montage.channel_positions
        # sequence_positions = ...
        # task_keys = ...
        # labels = ...
        # channel_mask = None
        # samples_mask = None

    async def on_shutdown(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        self._streamer.request_stop()

    def action_quit(self) -> None:
        self.exit()

    def on_select_changed(self, event: Select.Changed) -> None:  # type: ignore[name-defined]
        """Update y-limits for all time series plots when dropdown changes."""
        if getattr(event.select, "id", None) != "ylim-select":
            return
        value = event.value
        if value == "auto":
            new_ylim: Optional[float] = None
        else:
            try:
                new_ylim = float(value)
            except (TypeError, ValueError):
                new_ylim = None
        self._ylim_uv = new_ylim
        for plot in self._timeseries:
            plot.set_ylim(new_ylim)


def main() -> int:
    args = parse_args()

    try:
        config: AppConfig = load_app_config()
    except ValueError as err:
        logger.error("Invalid configuration: {}", err)
        return 1

    board: BoardShim
    streamer: StreamerProtocol
    board_details: BoardDetails

    try:
        montage = Montage.from_json(config.montage_path)
        channel_map_indices = [channel.index for channel in montage.channels]
        channel_names = [channel.reference_label for channel in montage.channels]
        if args.mock_eeg:
            num_channels = len(montage.channels)
            mock_sampling_rate = 125
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
                name=format_board_name(board_id),
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
            montage=montage,
            board_details=board_details,
            model_loader_config_path=args.model_loader_config_path,
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
