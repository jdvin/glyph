import os
import threading
import time
from typing import Optional, Protocol

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowError
from loguru import logger


def _fatal_daemon_failure(streamer_name: str, exc: BaseException) -> None:
    """Log and hard-exit when a streaming daemon dies."""
    logger.opt(exception=exc).critical(
        "{} streaming daemon crashed; terminating process.", streamer_name
    )
    os._exit(1)


class ChannelDataBuffer:
    def __init__(self, num_channels: int, eeg_channels: list[int], buffer_size: int):
        self.num_channels = num_channels
        self.eeg_channels = eeg_channels
        self.buffer_size = buffer_size
        self.buffer = np.zeros((num_channels, buffer_size), dtype=np.float64)
        self.filled = 0
        self._write_idx = 0
        self._total_written = 0

    def push(self, chunk: np.ndarray):
        assert (
            chunk.ndim == 2 and chunk.shape[0] == self.num_channels
        ), f"Expected ({self.num_channels},{chunk.shape[1]}) shape, got {chunk.shape}"
        T = chunk.shape[1]
        if T == 0:
            return
        if T >= self.buffer_size:
            chunk = chunk[:, -self.buffer_size :]
            T = self.buffer_size

        # Write into the circular buffer, handling wrap-around explicitly.
        first = min(T, self.buffer_size - self._write_idx)
        if first > 0:
            self.buffer[:, self._write_idx : self._write_idx + first] = chunk[:, :first]
        remaining = T - first
        if remaining > 0:
            self.buffer[:, :remaining] = chunk[:, first : first + remaining]

        self._write_idx = (self._write_idx + T) % self.buffer_size
        self.filled = min(self.buffer_size, self.filled + T)
        self._total_written += T

    def ready(self, T: int | None = None) -> bool:
        return self.prop_filled(T) == 1

    def prop_filled(self, T: int | None = None) -> float:
        return self.filled / (T or self.buffer_size)

    def get(self, T: int | None = None) -> np.ndarray:
        length = T or self.buffer_size
        result = np.zeros((self.num_channels, length), dtype=self.buffer.dtype)
        available = min(length, self.filled)
        if available == 0:
            return result

        recent = self._get_last_samples(available)
        result[:, -available:] = recent
        return result

    def get_eeg(self, T: int | None = None) -> np.ndarray:
        return self.get(T)[self.eeg_channels, :]

    def get_eeg_since(self, start_total: int) -> tuple[np.ndarray, int, bool]:
        """
        Return EEG samples that arrived after `start_total`.

        Args:
            start_total: Global sample index (monotonic) that marks the last
                processed sample. Samples that were overwritten before this
                index are skipped automatically.

        Returns:
            A tuple of (chunk, new_total, dropped) where:
              - chunk has shape (len(eeg_channels), T_new)
              - new_total is the latest total sample count
              - dropped is True if older samples were no longer available
        """
        chunk, new_total, dropped = self._get_new_samples(start_total)
        return chunk[self.eeg_channels, :], new_total, dropped

    def _get_new_samples(
        self, start_total: int
    ) -> tuple[np.ndarray, int, bool]:
        """Internal helper that returns raw channel data since a sample index."""
        current_total = self._total_written
        available_start = current_total - self.filled
        dropped = start_total < available_start
        start = max(start_total, available_start)
        new_count = current_total - start
        if new_count <= 0:
            empty = np.zeros(
                (self.num_channels, 0), dtype=self.buffer.dtype
            )
            return empty, current_total, dropped
        recent = self._get_last_samples(new_count)
        return recent, current_total, dropped

    def _get_last_samples(self, count: int) -> np.ndarray:
        assert 0 < count <= self.filled
        end = self._write_idx
        start = (end - count) % self.buffer_size
        if start < end:
            return self.buffer[:, start:end].copy()
        # Wrap-around: concatenate the tail and head slices
        return np.hstack((self.buffer[:, start:].copy(), self.buffer[:, :end].copy()))

    @property
    def total_samples_written(self) -> int:
        """Total number of samples that have been pushed into the buffer."""
        return self._total_written


class ChannelHealthMeter:
    """
    Streaming signal-quality meter for active EEG electrodes.
    - Maintains a rolling window.
    - Computes per-channel score (0–100) and flags:
      'hum', 'noise_high', 'noise_low', 'flatline', 'uncorrelated', 'shorted', 'rail'
    - Designed for ThinkPulse/active electrodes (no impedance gating).
    """

    def __init__(
        self,
        n_channels: int,
        fs: float,
        window_sec: float,
        mains_hz: float = 50.0,  # 60.0 in some regions
        hum_band_hz: float = 1.0,  # +/- width around mains & first harmonic
        brain_band: tuple = (1.0, 45.0),
        flatline_std_uv: float = 0.7,  # < ~0.7 µV over the window → likely flatline/short
        corr_low_thresh: float = 0.20,
        corr_shorted_thresh: float = 0.98,
        z_outlier_hi: float = 3.0,  # |z| > 3 in bandpower deviation → outlier
        rail_weight: float = 0.40,  # penalty per 100% railed (scale as needed)
    ):
        self.C = int(n_channels)
        self.fs = float(fs)
        self.N = int(round(window_sec * fs))
        self.window_sec = float(window_sec)

        self.mains_hz = float(mains_hz)
        self.hum_band_hz = float(hum_band_hz)
        self.brain_band = (float(brain_band[0]), float(brain_band[1]))

        self.flatline_std_uv = float(flatline_std_uv)
        self.corr_low_thresh = float(corr_low_thresh)
        self.corr_shorted_thresh = float(corr_shorted_thresh)
        self.z_outlier_hi = float(z_outlier_hi)
        self.rail_weight = float(rail_weight)

        # Optional aux metrics you can pass in (from your own detectors)
        self._adc_rail_pct = None  # shape (C,), 0..100
        self._plot_rail_pct = None  # shape (C,), 0..100

        # Precompute frequency grid for rFFT
        self._freqs = np.fft.rfftfreq(self.N, d=1.0 / self.fs)
        self._hann = np.hanning(self.N)

        # Index helpers
        def fmask(lo, hi):
            f = self._freqs
            return (f >= lo) & (f <= hi)

        self._brain_mask = fmask(*self.brain_band)
        # Keep brain band strictly >0 Hz to avoid DC
        self._brain_mask &= self._freqs > 0

        w = self.hum_band_hz
        self._hum1_mask = fmask(self.mains_hz - w, self.mains_hz + w)
        self._hum2_mask = fmask(2 * self.mains_hz - w, 2 * self.mains_hz + w)

    # ---------- streaming API ----------

    def compute(self, buffer: ChannelDataBuffer) -> tuple[np.ndarray, list[dict]]:
        """
        Returns:
          scores  : (C,) float in [0,100]
          flags   : list of dicts per channel with metrics & human-readable reasons
        """
        if not buffer.ready(self.N):
            # Not enough data yet; return neutral
            return np.full(self.C, 100.0), [
                {"reasons": ["warming_up"], "metrics": {}} for _ in range(self.C)
            ]

        X = buffer.get_eeg(self.N)

        # ---- time-domain prelims ----
        X -= X.mean(axis=1, keepdims=True)  # detrend (remove DC)
        std_uv = X.std(axis=1, ddof=0)  # overall variability (µV)

        # ---- frequency-domain (Hann + rFFT) ----
        W = self._hann
        # normalize window power so band sums are comparable across chunks
        win_norm = np.sum(W**2)
        F = np.fft.rfft(X * W, axis=1)
        P = (np.abs(F) ** 2) / max(win_norm, 1e-12)  # power spectrum (arbitrary scale)

        # Band powers
        def bandpow(mask):  # (C,)
            if np.count_nonzero(mask) == 0:
                return np.zeros(self.C)
            return P[:, mask].sum(axis=1)

        P_brain = bandpow(self._brain_mask) + 1e-18  # avoid /0
        P_hum = bandpow(self._hum1_mask) + bandpow(self._hum2_mask)

        # Line-Hum Ratio (dB)
        hum_db = 10.0 * np.log10(P_hum / P_brain)  # more negative is better

        # Robust deviation: z-score of log bandpower relative to channel-median
        logP = np.log(P_brain)
        med = np.median(logP)
        mad = np.median(np.abs(logP - med)) + 1e-12
        z_dev = 0.6745 * (logP - med) / mad  # ≈ z from MAD

        # Correlation to robust reference (median across channels)
        ref = np.median(X, axis=0, keepdims=True)  # (1,N)
        # prevent zero-variance issues
        Xc = X - X.mean(axis=1, keepdims=True)
        refc = ref - ref.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(Xc, axis=1) * np.linalg.norm(refc, axis=1)[0] + 1e-12
        corr = (Xc @ refc.T)[:, 0] / denom  # (C,)

        # Optional: incorporate rail percentages if supplied
        rail_pct = None
        if self._adc_rail_pct is not None:
            rail_pct = np.clip(self._adc_rail_pct, 0, 100)
        elif self._plot_rail_pct is not None:
            rail_pct = np.clip(self._plot_rail_pct, 0, 100)

        # ---- scoring ----
        scores = np.full(self.C, 100.0, dtype=float)
        flags = []

        # Thresholds tuned for active electrodes
        hum_thresh_db = -15.0  # if > -15 dB, mains is too dominant
        # penalties (max contributions)
        max_pen_hum = 30.0
        max_pen_dev = 25.0
        max_pen_flat = 40.0
        max_pen_corr = 30.0
        max_pen_rail = 40.0

        for c in range(self.C):
            reasons = []
            metrics = dict(
                hum_db=float(hum_db[c]),
                z_dev=float(z_dev[c]),
                std_uv=float(std_uv[c]),
                corr=float(corr[c]),
            )

            # 1) Hum penalty
            if hum_db[c] > hum_thresh_db:
                # linear ramp from -15 dB (0) to +0 dB (max)
                pen = max_pen_hum * min(1.0, (hum_db[c] - hum_thresh_db) / 15.0)
                scores[c] -= pen
                reasons.append(f"hum ({hum_db[c]:.1f} dB)")

            # 2) Deviation penalty (too high or too low bandpower)
            if abs(z_dev[c]) > (self.z_outlier_hi - 1.0):  # soften edge a bit
                pen = max_pen_dev * min(1.0, (abs(z_dev[c]) - 2.0) / 2.0)
                scores[c] -= max(0.0, pen)
                if z_dev[c] > 0:
                    reasons.append(f"noise_high (z={z_dev[c]:.1f})")
                else:
                    reasons.append(f"noise_low (z={z_dev[c]:.1f})")

            # 3) Flatline (very low time-domain std)
            if std_uv[c] < self.flatline_std_uv:
                scores[c] -= max_pen_flat
                reasons.append(f"flatline (std={std_uv[c]:.2f} µV)")

            # 4) Correlation checks
            if corr[c] < self.corr_low_thresh:
                # worse correlation → larger penalty
                pen = max_pen_corr * min(
                    1.0, (self.corr_low_thresh - corr[c]) / self.corr_low_thresh
                )
                scores[c] -= max(0.0, pen)
                reasons.append(f"uncorrelated (r={corr[c]:.2f})")
            elif corr[c] > self.corr_shorted_thresh and std_uv[c] < 2.0:
                # suspiciously identical & tiny variance → possible short/duplicate
                scores[c] -= 0.5 * max_pen_corr
                reasons.append(f"shorted_like (r={corr[c]:.2f})")

            # 5) Rail penalty (if provided)
            if rail_pct is not None:
                pen = max_pen_rail * (rail_pct[c] / 100.0) * self.rail_weight
                if pen > 0:
                    scores[c] -= pen
                    reasons.append(f"rail ({rail_pct[c]:.0f}%)")

            # finalize
            scores[c] = float(np.clip(scores[c], 0.0, 100.0))
            flags.append({"reasons": reasons or ["ok"], "metrics": metrics})

        return scores, flags


class StreamerProtocol(Protocol):
    buffer: ChannelDataBuffer
    health_metrics: ChannelHealthMeter
    sampling_rate: int

    def start(self) -> None: ...

    def request_stop(self) -> None: ...

    def close(self) -> None: ...


class BrainFlowStreamer:
    """Background thread that feeds BrainFlow data into a queue for the UI."""

    def __init__(
        self, board: BoardShim, buffer_size: int, poll_interval: float
    ) -> None:
        self._board = board
        self._board_id = board.get_board_id()
        self._eeg_channels = board.get_eeg_channels(self._board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self._board_id)
        self._buffer_size = buffer_size
        self._poll_interval = poll_interval
        # 4 channels for accelerometers and sample index.
        # No idea what the remaining 12 are for.
        self.buffer = ChannelDataBuffer(
            len(self._eeg_channels) + 16, self._eeg_channels, buffer_size
        )
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stream_started = False
        self.health_metrics = ChannelHealthMeter(
            len(self._eeg_channels),
            self.sampling_rate,
            window_sec=buffer_size / self.sampling_rate,
        )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._board.prepare_session()
        self._board.start_stream()
        self._stream_started = True

        def _run() -> None:
            logger.info("BrainFlow streaming thread started.")
            try:
                while not self._stop_event.is_set():
                    try:
                        chunk = self._board.get_board_data(self._buffer_size)
                        if chunk.size > 0:
                            self.buffer.push(chunk)
                        else:
                            time.sleep(self._poll_interval)
                    except BrainFlowError as err:
                        _fatal_daemon_failure("BrainFlow", err)
            except Exception as exc:
                _fatal_daemon_failure("BrainFlow", exc)
            finally:
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
        sampling_rate: int = 125,
        noise_scale: float = 5.0,
    ) -> None:
        self._num_channels = num_channels
        self._buffer_size = buffer_size
        self._poll_interval = poll_interval
        self.sampling_rate = sampling_rate
        self._noise_scale = noise_scale
        self.buffer = ChannelDataBuffer(
            num_channels + 1, list(range(num_channels)), buffer_size
        )
        self.sampling_rate = sampling_rate
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._rng = np.random.default_rng()
        self._sample_index = 0
        self._base_frequencies = np.linspace(8.0, 15.0, num_channels)

        self.health_metrics = ChannelHealthMeter(
            num_channels,
            sampling_rate,
            window_sec=5.0,
        )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        def _run() -> None:
            logger.info("Mock EEG streaming thread started.")
            try:
                while not self._stop_event.is_set():
                    chunk = self._generate_chunk()
                    self.buffer.push(chunk)
                    time.sleep(self._poll_interval)
            except Exception as exc:
                _fatal_daemon_failure("Mock EEG", exc)
            finally:
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
        chunk_size = int(self._poll_interval * self.sampling_rate)
        buffer_index = np.arange(chunk_size, dtype=np.float64)
        t = (buffer_index + self._sample_index) / self.sampling_rate
        self._sample_index += chunk_size
        signals = [buffer_index]
        for idx, freq in enumerate(self._base_frequencies):
            phase = idx * np.pi / 4
            amplitude = 40.0 + 5.0 * idx
            baseline = 10.0 * np.sin(2 * np.pi * 1.0 * t)
            alpha_wave = amplitude * np.sin(2 * np.pi * freq * t + phase)
            noise = self._rng.normal(0.0, self._noise_scale, size=chunk_size)
            signals.append(alpha_wave + baseline + noise)
        return np.vstack(signals)
