from __future__ import annotations

import io
import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image
from pydantic import BaseModel

from src.capture.triggers import process_frame

logger = logging.getLogger(__name__)

router = APIRouter()


class IngestResponse(BaseModel):
    frame_id: int


@router.post("/capture/ingest", response_model=IngestResponse)
async def ingest_frame(
    request: Request,
    image: UploadFile = File(...),
    app_name: str = Form(""),
    window_name: str = Form(""),
    trigger: str = Form("manual"),
) -> IngestResponse:
    """Receive a captured frame from the Swift overlay and process it."""
    state = request.app.state

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image")

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    frame_id = await process_frame(
        image=pil_image,
        app_name=app_name or None,
        window_name=window_name or None,
        trigger=trigger,
        db=state.db,
        writer=state.snapshot_writer,
        providers=state.context_providers,
        general_agent=state.general_agent,
        browser_agent=state.browser_agent,
    )

    logger.debug("Ingested frame %d from overlay (trigger=%s, app=%s)", frame_id, trigger, app_name)
    return IngestResponse(frame_id=frame_id)
