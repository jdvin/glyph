"""Interactive TUI for streaming OpenBCI Cyton-Daisy data with BrainFlow (line plots, proper µV axes)."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional
import json

import numpy as np
from numpy.lib.format import open_memmap
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
from .streamer import (
    BrainFlowStreamer,
    ChannelDataBuffer,
    MockEEGStreamer,
    StreamerProtocol,
)
from .utils import (
    AppConfig,
    FilterConfig,
    ModelLoaderConfig,
    Montage,
    create_board,
    detect_serial_port,
    load_app_config,
    load_css,
    format_board_name,
    load_model,
    per_channel_median_shift,
    per_channel_normalize,
    StreamingSOSFilter,
)


def _positive_int(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected integer, received '{value}'."
        ) from exc
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return ivalue


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
    parser.add_argument(
        "--data-task",
        default="",
        help="Optional task label to store alongside each saved EEG frame.",
    )
    parser.add_argument(
        "--data-label",
        default="",
        help="Optional class label to store alongside each saved EEG frame.",
    )
    parser.add_argument(
        "--memmap-path",
        default=None,
        help=(
            "Base path for NumPy memmap files; creates {path}_eeg.npy and "
            "{path}_labels.npy to store streamed frames."
        ),
    )
    parser.add_argument(
        "--memmap-frames",
        type=_positive_int,
        default=5,
        help="Number of frames (N) to keep inside the memmap ring buffer (default: 5).",
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
        filter_configs: list[FilterConfig],
        model_loader_config_path: str | None,
        memmap_path: str | None = None,
        memmap_frames: int = 5,
        data_task: str = "",
        data_label: str = "",
        css: str = "",
    ) -> None:
        super().__init__()
        # Set CSS dynamically
        if css:
            self.CSS = css
        self._streamer = streamer
        self._channel_indices = channel_indices
        self._channel_names = channel_names
        self._filtered_channel_count = len(self._channel_names)
        self._filtered_channel_indices = list(
            range(self._filtered_channel_count)
        )
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
        self._memmap_path = memmap_path
        self._memmap_frames = memmap_frames
        self._data_task = data_task
        self._data_label = data_label
        self._eeg_memmap: np.memmap | None = None
        self._labels_memmap: np.memmap | None = None
        self._memmap_index = 0
        self._buffer_size = getattr(self._streamer.buffer, "buffer_size", None)
        self._filter = StreamingSOSFilter(
            filter_configs, self._streamer.sampling_rate, len(self._channel_names)
        )
        self._filtered_buffer: ChannelDataBuffer | None = None
        self._last_filtered_sample = 0
        if self._buffer_size is not None:
            self._filtered_buffer = ChannelDataBuffer(
                self._filtered_channel_count,
                self._filtered_channel_indices,
                self._buffer_size,
            )
        else:
            logger.warning(
                "Streaming buffer size unavailable; filtering will reprocess the entire window."
            )
        self._model = None
        self._model_config = None
        if model_loader_config_path:
            logger.info("Loading model from {}", model_loader_config_path)
            with open(model_loader_config_path, "r", encoding="utf-8") as file:
                self._model_loader_config = ModelLoaderConfig(**json.load(file))
            self._model, self._model_config = load_model(self._model_loader_config)
        self._model_details_panel = ModelDetailsPanel(self._model)
        self._model_probs_panel = ModelProbsPlot(
            self._model, self._model_config.labels_map if self._model_config else {}
        )
        if self._memmap_path and self._buffer_size is not None:
            self._initialize_memmaps(len(self._channel_names))
        elif self._memmap_path:
            logger.warning(
                "Memmap path {} ignored because buffer size could not be determined.",
                self._memmap_path,
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

    def _initialize_memmaps(self, num_channels: int) -> None:
        assert self._buffer_size is not None
        eeg_path = f"{self._memmap_path}_eeg.npy"
        labels_path = f"{self._memmap_path}_labels.npy"
        try:
            self._eeg_memmap = open_memmap(
                eeg_path,
                dtype=np.float32,
                mode="w+",
                shape=(
                    self._memmap_frames,
                    num_channels,
                    self._buffer_size,
                ),
            )
            logger.info(
                "Initialized EEG memmap at {} with shape ({}, {}, {}).",
                eeg_path,
                self._memmap_frames,
                num_channels,
                self._buffer_size,
            )
            max_label_chars = max(16, len(self._data_task), len(self._data_label), 1)
            label_dtype = np.dtype(f"U{max_label_chars}")
            self._labels_memmap = open_memmap(
                labels_path,
                dtype=label_dtype,
                mode="w+",
                shape=(self._memmap_frames, 2),
            )
            self._labels_memmap[:, 0] = self._data_task
            self._labels_memmap[:, 1] = self._data_label
            self._labels_memmap.flush()
            logger.info(
                "Initialized label memmap at {} with shape ({}, 2).",
                labels_path,
                self._memmap_frames,
            )
        except (OSError, ValueError) as exc:
            logger.error(
                "Failed to create memmap files at base {}: {}", self._memmap_path, exc
            )
            self._eeg_memmap = None
            self._labels_memmap = None

    def _get_filtered_eeg(self) -> np.ndarray:
        """Return the latest filtered EEG window, filtering only newly arrived samples."""
        buffer = self._streamer.buffer
        if self._filtered_buffer is None or not hasattr(
            buffer, "get_eeg_since"
        ):
            raw_window = buffer.get_eeg()
            return self._filter.process(raw_window)

        chunk, total, dropped = buffer.get_eeg_since(self._last_filtered_sample)
        if dropped:
            self._filter.reset()
            self._filtered_buffer = ChannelDataBuffer(
                self._filtered_channel_count,
                self._filtered_channel_indices,
                self._buffer_size,
            )
        if chunk.size:
            filtered_chunk = self._filter.process(chunk)
            self._filtered_buffer.push(filtered_chunk)
            self._last_filtered_sample = total
        return self._filtered_buffer.get()

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
        eeg_data = self._get_filtered_eeg()

        for idx, row in enumerate(per_channel_median_shift(eeg_data)):
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
        self._write_memmap_frame(eeg_data)
        # Run model inference if model is loaded
        if self._model is not None:
            device = self._model_loader_config.device
            channel_signals = torch.tensor(
                per_channel_normalize(eeg_data), device=device, dtype=torch.float32
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
                channel_signals,
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

    def _write_memmap_frame(self, eeg_frame: np.ndarray) -> None:
        if self._eeg_memmap is None:
            return
        expected_shape = self._eeg_memmap[self._memmap_index].shape
        if eeg_frame.shape != expected_shape:
            logger.warning(
                "Skipping memmap write because frame shape {} does not match expected {}.",
                eeg_frame.shape,
                expected_shape,
            )
            return
        np.copyto(
            self._eeg_memmap[self._memmap_index],
            eeg_frame.astype(np.float32, copy=False),
            casting="unsafe",
        )
        self._eeg_memmap.flush()
        if self._labels_memmap is not None:
            self._labels_memmap[self._memmap_index, 0] = self._data_task
            self._labels_memmap[self._memmap_index, 1] = self._data_label
            self._labels_memmap.flush()
        self._memmap_index = (self._memmap_index + 1) % self._memmap_frames

    async def on_shutdown(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        self._streamer.request_stop()
        if self._eeg_memmap is not None:
            self._eeg_memmap.flush()
        if self._labels_memmap is not None:
            self._labels_memmap.flush()

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
            memmap_path=args.memmap_path,
            memmap_frames=args.memmap_frames,
            filter_configs=config.filter_configs,
            data_task=args.data_task,
            data_label=args.data_label,
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
