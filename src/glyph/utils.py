from dataclasses import dataclass
from importlib import resources
from enum import Enum
import inspect
import json
import mne
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from brainflow.board_shim import (
    BoardIds,
    BoardShim,
    BrainFlowInputParams,
)
from loguru import logger
import numpy as np
from serial.tools import list_ports
import torch
from torch import nn
from torch.package.package_importer import PackageImporter
from scipy import signal


class FilterType(Enum):
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"


@dataclass(frozen=True)
class FilterConfig:
    bounds: tuple[float, float]
    btype: FilterType


@dataclass(frozen=True)
class AppConfig:
    buffer_size: int
    poll_interval: float
    window_size: int
    refresh_interval: float
    ylim: Optional[float]
    montage_path: str
    css_path: str
    filter_configs: list[FilterConfig]


@dataclass(frozen=True)
class BoardDetails:
    source: str
    name: str
    board_id: Optional[int]
    sampling_rate_hz: Optional[float]
    eeg_channel_count: Optional[int]
    serial_port: Optional[str] = None


@dataclass(frozen=True)
class Channel:
    index: int
    board_label: str
    reference_label: str


class Montage:
    def __init__(self, reference_system: str, channels: list[Channel]):
        self.reference_system = reference_system
        self.channels = channels
        montage = mne.channels.make_standard_montage(
            self.reference_system
        ).get_positions()["ch_pos"]
        self.channel_positions = np.vstack([
            montage[ch.reference_label] for ch in self.channels
        ])

    @classmethod
    def from_json(cls, path: str) -> "Montage":
        with open(
            os.path.join("src", "glyph", "data", path), "r", encoding="utf-8"
        ) as file:
            config_data = json.load(file)
        reference_system = config_data["reference_system"]
        assert reference_system in ("standard_1020", "standard_1005")
        channels = [Channel(**channel) for channel in config_data["channels"]]
        return cls(reference_system=reference_system, channels=channels)


@dataclass
class ElectrodeStyle:
    cmap: str = "plasma"  # plotext colormap name (fallback handled)
    radius: float = 0.98  # draw a head circle at this radius
    label_color: str = "white"
    label_shift: tuple[float, float] = (0.0, 0.0)  # nudge labels (dx, dy)
    show_head_circle: bool = True


@dataclass(frozen=True)
class ModelLoaderConfig:
    """Configuration for a model."""

    package_path: str
    model_name: str
    device: str


class StreamingSOSFilter:
    def __init__(self, filter_configs: list[FilterConfig], fs: int, n_channels: int):
        self.sos = [
            signal.butter(4, fc.bounds, btype=fc.btype.value, fs=fs, output="sos")
            for fc in filter_configs
        ]
        zi_base = [signal.sosfilt_zi(sos_i) for sos_i in self.sos]  # (n_sections, 2)
        # add channel dimension
        self.zi = [
            np.repeat(zi_base[:, np.newaxis, :], n_channels, axis=-2)
            for zi_base in zi_base
        ]

    def process(self, x: np.ndarray) -> np.ndarray:
        # x shape: (n_channels, n_samples)
        y = x.copy()
        for i in range(len(self.sos)):
            y, self.zi[i] = signal.sosfilt(self.sos[i], y, zi=self.zi[i], axis=-1)
        return x


def load_app_config(config_path: Optional[str] = None) -> AppConfig:
    """Load application configuration from JSON."""
    if config_path:
        path = Path(config_path).expanduser()
        with path.open("r", encoding="utf-8") as file:
            config_data = json.load(file)
        logger.info("Loaded configuration overrides from {}", path)
    else:
        resource = resources.files("glyph.config").joinpath("defaults.json")
        with resource.open("r", encoding="utf-8") as file:
            config_data = json.load(file)
        logger.debug("Loaded bundled configuration defaults.")

    return _parse_config(config_data)


def load_css(css_path: str) -> str:
    """Load CSS from a file path.

    Args:
        css_path: Relative path to CSS file (relative to glyph.config)

    Returns:
        CSS content as a string
    """
    try:
        resource = resources.files("glyph.config").joinpath(css_path)
        with resource.open("r", encoding="utf-8") as file:
            css_content = file.read()
        logger.debug(f"Loaded CSS from {css_path}")
        return css_content
    except Exception as e:
        logger.warning(f"Could not load CSS from {css_path}: {e}. Using empty CSS.")
        return ""


def _parse_config(config_data: dict[str, Any]) -> AppConfig:
    try:
        buffer_size = int(config_data["buffer_size"])
        poll_interval = float(config_data["poll_interval"])
        window_size = int(config_data["window_size"])
        refresh_interval = float(config_data["refresh_interval"])
        montage_path = config_data["montage_path"]
        css_path = config_data["css_path"]
        filter_configs = [
            FilterConfig(fc["bounds"], FilterType(fc["btype"]))
            for fc in config_data["filter_configs"]
        ]
    except KeyError as missing:
        raise ValueError(
            f"Missing required config key: {missing.args[0]!s}"
        ) from missing
    except (TypeError, ValueError) as err:
        raise ValueError("Invalid numeric value in configuration.") from err

    ylim_value = config_data.get("ylim", None)
    if ylim_value is None:
        ylim: Optional[float] = None
    else:
        try:
            ylim = float(ylim_value)
        except (TypeError, ValueError) as err:
            raise ValueError("Config value 'ylim' must be numeric or null.") from err

    return AppConfig(
        buffer_size=buffer_size,
        poll_interval=poll_interval,
        window_size=window_size,
        refresh_interval=refresh_interval,
        ylim=ylim,
        montage_path=montage_path,
        css_path=css_path,
        filter_configs=filter_configs,
    )


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


def format_board_name(board_id: int) -> str:
    try:
        enum_name = BoardIds(board_id).name
    except ValueError:
        return f"Board {board_id}"
    return enum_name.replace("_", " ").title()


def describe_function(func):
    """Return a string containing a function's signature and docstring."""
    sig = str(inspect.signature(func))
    doc = inspect.getdoc(func) or "(no docstring)"
    return f"{func.__name__}{sig}\n\n{doc}"


def load_model(
    model_loader_config: ModelLoaderConfig,
) -> tuple[nn.Module, dict[int, str]]:
    """Load a PyTorch model from a pytorch package.

    Returns:
        A tuple of (model, labels_map) where labels_map maps class indices to labels.
    """

    imp = PackageImporter(model_loader_config.package_path)
    # Assumes module name is the model name in lower case.
    model_module = f"src.{model_loader_config.model_name.lower()}"
    Model = getattr(
        imp.import_module(model_module),
        model_loader_config.model_name,
    )
    state = imp.load_pickle("assets", "state.pkl")
    model_config = imp.load_pickle("config", "model_config.pkl")

    # Extract labels_map from model_config
    labels_map: dict[int, str] = {}
    try:
        labels_map = model_config.tasks[0].labels_map
        logger.info(f"Loaded labels_map with {len(labels_map)} classes")
    except (AttributeError, IndexError) as e:
        logger.warning(
            f"Could not load labels_map from model_config.tasks[0].labels_map: {e}"
        )
        labels_map = {}

    model = Model(model_config, rank=0, world_size=1).eval()
    model.load_state_dict(state)
    model.to(model_loader_config.device).eval()
    return model, model_config


def per_channel_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize each channel of a tensor independently."""
    return (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)
