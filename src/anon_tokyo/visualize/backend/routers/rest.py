from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ..deps import get_service
from ..services.visualizer import WebVisualizerService


router = APIRouter(tags=["rest"])


@router.get("/env")
def fetch_env(svc: WebVisualizerService = Depends(get_service)) -> dict[str, Any]:
    try:
        return svc.fetch_env()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/files")
def fetch_files(svc: WebVisualizerService = Depends(get_service)) -> dict[str, Any]:
    try:
        return svc.fetch_files()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/env")
def initialize_env(payload: dict[str, Any], svc: WebVisualizerService = Depends(get_service)) -> dict[str, Any]:
    try:
        return svc.initialize_env(**payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/batch/{batch_idx}")
def fetch_batch(
    batch_idx: int,
    batch_size: int | None = Query(None),
    svc: WebVisualizerService = Depends(get_service),
) -> dict[str, Any]:
    try:
        return svc.fetch_batch(batch_idx, batch_size=batch_size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/batch/{batch_idx}/world/{world_idx}")
def fetch_world(
    batch_idx: int,
    world_idx: int,
    svc: WebVisualizerService = Depends(get_service),
) -> dict[str, Any]:
    try:
        return svc.fetch_world(batch_idx, world_idx)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/batch/{batch_idx}/world/{world_idx}/rollout")
def rollout_world(
    batch_idx: int,
    world_idx: int,
    payload: dict[str, Any],
    svc: WebVisualizerService = Depends(get_service),
) -> dict[str, Any]:
    try:
        return svc.rollout_world(batch_idx, world_idx, count=payload.get("count"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
