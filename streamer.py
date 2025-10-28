import threading
import time
from queue import Queue
from typing import Optional, Protocol

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowError
from loguru import logger
from brainflow.data_filter import (
    DataFilter,
    DetrendOperations,
    FilterTypes,
)


class StreamerProtocol(Protocol):
    queue: Queue[np.ndarray]

    def start(self) -> None:
        ...

    def request_stop(self) -> None:
        ...

    def close(self) -> None:
        ...


class BrainFlowStreamer:
    """Background thread that feeds BrainFlow data into a queue for the UI."""

    def __init__(
        self, board: BoardShim, buffer_size: int, poll_interval: float
    ) -> None:
        self._board = board
        self._board_id = board.get_board_id()
        self._eeg_channels = board.get_eeg_channels(self._board_id)
        self._sampling_rate = BoardShim.get_sampling_rate(self._board_id)
        self._buffer_size = buffer_size
        self._poll_interval = poll_interval
        self.queue: Queue[np.ndarray] = Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stream_started = False

    def filter_transform(self, data: np.ndarray) -> np.ndarray:
        """Filter and transform data from the board."""
        for count, channel in enumerate(self._eeg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(
                data[channel],
                self._sampling_rate,
                3.0,
                45.0,
                2,
                FilterTypes.BUTTERWORTH_ZERO_PHASE,
                0,
            )
            DataFilter.perform_bandstop(
                data[channel],
                self._sampling_rate,
                48.0,
                52.0,
                2,
                FilterTypes.BUTTERWORTH_ZERO_PHASE,
                0,
            )
            DataFilter.perform_bandstop(
                data[channel],
                self._sampling_rate,
                58.0,
                62.0,
                2,
                FilterTypes.BUTTERWORTH_ZERO_PHASE,
                0,
            )
        return data

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
                        self.queue.put(self.filter_transform(chunk))
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


class MockEEGStreamer:
    """Background thread that feeds simulated EEG data into a queue for the UI."""

    def __init__(
        self,
        num_channels: int,
        buffer_size: int,
        poll_interval: float,
        sampling_rate: float = 250.0,
        noise_scale: float = 5.0,
    ) -> None:
        self._num_channels = num_channels
        self._buffer_size = buffer_size
        self._poll_interval = poll_interval
        self._sampling_rate = sampling_rate
        self._noise_scale = noise_scale
        self.queue: Queue[np.ndarray] = Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._rng = np.random.default_rng()
        self._sample_index = 0
        self._base_frequencies = np.linspace(8.0, 15.0, num_channels)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        def _run() -> None:
            logger.info("Mock EEG streaming thread started.")
            while not self._stop_event.is_set():
                chunk = self._generate_chunk()
                self.queue.put(chunk)
                time.sleep(self._poll_interval)
            logger.info("Mock EEG streaming thread stopped.")

        self._thread = threading.Thread(
            target=_run, name="MockEEGStreamer", daemon=True
        )
        self._thread.start()

    def request_stop(self) -> None:
        self._stop_event.set()

    def close(self) -> None:
        self.request_stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    @property
    def num_channels(self) -> int:
        return self._num_channels

    def _generate_chunk(self) -> np.ndarray:
        t = (
            np.arange(self._buffer_size, dtype=np.float64) + self._sample_index
        ) / self._sampling_rate
        self._sample_index += self._buffer_size

        signals = []
        for idx, freq in enumerate(self._base_frequencies):
            phase = idx * np.pi / 4
            amplitude = 40.0 + 5.0 * idx
            baseline = 10.0 * np.sin(2 * np.pi * 1.0 * t)
            alpha_wave = amplitude * np.sin(2 * np.pi * freq * t + phase)
            noise = self._rng.normal(0.0, self._noise_scale, size=self._buffer_size)
            signals.append(alpha_wave + baseline + noise)
        return np.vstack(signals)
