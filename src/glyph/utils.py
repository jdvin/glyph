from typing import Iterable, Optional

from brainflow.board_shim import (
    BoardIds,
    BoardShim,
    BrainFlowInputParams,
)
from loguru import logger
from serial.tools import list_ports


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
