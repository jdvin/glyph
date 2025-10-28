import threading
import time
from queue import Queue
from typing import Optional

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowError
from loguru import logger
from brainflow.data_filter import (
    DataFilter,
    DetrendOperations,
    FilterTypes,
)


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
