from __future__ import annotations

from .services.visualizer import WebVisualizerService


_SERVICE = WebVisualizerService.from_env()


def get_service() -> WebVisualizerService:
    return _SERVICE

