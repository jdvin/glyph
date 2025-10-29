from dataclasses import dataclass
from importlib import resources
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from brainflow.board_shim import (
    BoardIds,
    BoardShim,
    BrainFlowInputParams,
)
from loguru import logger
from serial.tools import list_ports


@dataclass(frozen=True)
class AppConfig:
    buffer_size: int
    poll_interval: float
    window_size: int
    refresh_interval: float
    ylim: Optional[float]
    montage_path: str


@dataclass(frozen=True)
class Channel:
    index: int
    board_label: str
    reference_label: str


@dataclass(frozen=True)
class Montage:
    reference_system: str
    channel_map: list[Channel]

    @classmethod
    def from_json(cls, path: str) -> "Montage":
        with open(
            os.path.join("src", "glyph", "data", path), "r", encoding="utf-8"
        ) as file:
            config_data = json.load(file)
        reference_system = config_data["reference_system"]
        assert reference_system in ("standard_1020", "standard_1005")
        channel_map = [Channel(**channel) for channel in config_data["channel_map"]]
        return cls(reference_system=reference_system, channel_map=channel_map)


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


def _parse_config(config_data: dict[str, Any]) -> AppConfig:
    try:
        buffer_size = int(config_data["buffer_size"])
        poll_interval = float(config_data["poll_interval"])
        window_size = int(config_data["window_size"])
        refresh_interval = float(config_data["refresh_interval"])
        montage_path = config_data["montage_path"]
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
