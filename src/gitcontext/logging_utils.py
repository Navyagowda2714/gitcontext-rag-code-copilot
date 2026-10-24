"""Structured (key=value) logging setup."""

from __future__ import annotations

import logging
import sys


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"ts={self.formatTime(record, '%Y-%m-%dT%H:%M:%S')} level={record.levelname.lower()} "
            f'logger={record.name} msg="{record.getMessage()}"'
        )
        if record.exc_info:
            base += f' exc="{self.formatException(record.exc_info)!r}"'
        return base


def configure_logging(level: int | str = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(KeyValueFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
