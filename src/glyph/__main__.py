"""Interactive TUI for streaming OpenBCI Cyton-Daisy data with BrainFlow (line plots, proper µV axes)."""

from __future__ import annotations

import argparse
import sys
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
import torch
from torch.nn import functional as F

from .plot import (
    ChannelLinePlot,
    GlyphicMap,
    BoardDetailsPanel,
    HealthMetricsPanel,
    BoardDetails,
    ModelDetailsPanel,
    ModelProbsPlot,
)
from .streamer import BrainFlowStreamer, MockEEGStreamer, StreamerProtocol
from .utils import (
    AppConfig,
    ModelLoaderConfig,
    Montage,
    create_board,
    detect_serial_port,
    load_app_config,
    load_css,
    format_board_name,
    load_model,
    per_channel_detrend,
    per_channel_mains_bandstop,
    per_channel_normalize,
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
        css: str = "",
    ) -> None:
        super().__init__()
        # Set CSS dynamically
        if css:
            self.CSS = css
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
        self._model = None
        self._labels_map = {}
        self._model_loader_config = None
        if model_loader_config_path:
            logger.info("Loading model from {}", model_loader_config_path)
            with open(model_loader_config_path, "r", encoding="utf-8") as file:
                self._model_loader_config = ModelLoaderConfig(**json.load(file))
            self._model, self._model_config = load_model(self._model_loader_config)
        self._model_details_panel = ModelDetailsPanel(self._model)
        self._model_probs_panel = ModelProbsPlot(
            self._model, self._model_config.labels_map
        )
        self._ts_scale = Select(
            options=[
                ("200 µV", "200"),
                ("100 µV", "100"),
                ("50 µV", "50"),
                ("Auto", "auto"),
            ],
            value=("auto" if self._ylim_uv is None else str(int(self._ylim_uv))),
            id="ylim-select",
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent():
            with TabPane("Channel Map"):
                with Container(id="channel-panel"):
                    yield self._board_panel
                    yield self._channel_map
                    yield self._health_panel
            with TabPane("Time Series"):
                # Y-limit selector above the plots
                yield self._ts_scale
                with Grid(id="timeseries-panel"):
                    for plot in self._timeseries:
                        yield plot
            with TabPane("Model"):
                with Container(id="model-panel"):
                    yield self._model_details_panel
                    yield self._model_probs_panel
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
        plot_data = per_channel_detrend(eeg_data, self._streamer.sampling_rate)
        eeg_data = per_channel_mains_bandstop(eeg_data, self._streamer.sampling_rate)
        plot_data = per_channel_mains_bandstop(plot_data, self._streamer.sampling_rate)
        for idx, row in enumerate(plot_data):
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

        # Run model inference if model is loaded
        if self._model is not None:
            device = self._model_loader_config.device
            channel_signals = torch.tensor(
                eeg_data, device=device, dtype=torch.float32
            ).unsqueeze(0)
            channel_positions = torch.tensor(
                self._montage.channel_positions, device=device, dtype=torch.float32
            ).unsqueeze(0)
            dc = self._model_config.data_config
            sequence_positions = torch.linspace(
                0,
                dc.sequence_length_seconds * dc.position_index_per_second,
                int(self._streamer.sampling_rate * dc.sequence_length_seconds),
                device=device,
            ).unsqueeze(0)
            task_keys = torch.tensor([[0]], device=device)
            labels = torch.tensor([0], device=device)
            samples_mask = torch.ones_like(sequence_positions)
            _, logits, _ = self._model(
                per_channel_normalize(channel_signals),
                channel_positions,
                sequence_positions,
                task_keys,
                labels,
                None,
                samples_mask,
            )
            # Compute softmax probabilities
            probs = F.softmax(logits, dim=-1)

            # Update model panel with probabilities
            probs_np = probs.cpu().detach().numpy().squeeze()
            self._model_probs_panel.update_probabilities(probs_np)

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

    # Load CSS from config
    css = load_css(config.css_path)

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
            css=css,
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
