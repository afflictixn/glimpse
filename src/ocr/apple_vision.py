from __future__ import annotations

import json
import logging

from PIL import Image

from src.storage.models import OCRResult

logger = logging.getLogger(__name__)


def perform_ocr(image: Image.Image) -> OCRResult:
    from Vision import VNImageRequestHandler, VNRecognizeTextRequest
    from Quartz import (
        CGColorSpaceCreateDeviceGray,
        CGDataProviderCreateWithData,
        CGImageCreate,
        kCGBitmapByteOrderDefault,
        kCGImageAlphaNone,
    )

    gray = image.convert("L")
    width, height = gray.size
    raw_data = gray.tobytes()

    color_space = CGColorSpaceCreateDeviceGray()
    provider = CGDataProviderCreateWithData(None, raw_data, len(raw_data), None)

    cg_image = CGImageCreate(
        width,
        height,
        8,  # bits per component
        8,  # bits per pixel
        width,  # bytes per row
        color_space,
        kCGBitmapByteOrderDefault | kCGImageAlphaNone,
        provider,
        None,  # decode array
        False,  # should interpolate
        0,  # rendering intent
    )

    if cg_image is None:
        logger.error("Failed to create CGImage for OCR")
        return OCRResult()

    handler = VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
    request = VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(1)  # accurate
    request.setUsesLanguageCorrection_(False)

    success, error = handler.performRequests_error_([request], None)
    if not success:
        logger.error("OCR request failed: %s", error)
        return OCRResult()

    results = request.results()
    if not results:
        return OCRResult()

    lines = []
    total_confidence = 0.0
    text_parts = []

    for observation in results:
        candidate = observation.topCandidates_(1)
        if not candidate:
            continue
        top = candidate[0]
        text = top.string()
        confidence = top.confidence()

        bbox = observation.boundingBox()
        x = bbox.origin.x
        y = 1.0 - bbox.origin.y - bbox.size.height  # flip to top-left origin
        w = bbox.size.width
        h = bbox.size.height

        lines.append({
            "text": text,
            "confidence": round(confidence, 4),
            "bbox": {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)},
        })
        text_parts.append(text)
        total_confidence += confidence

    full_text = "\n".join(text_parts)
    avg_confidence = total_confidence / len(lines) if lines else 0.0

    return OCRResult(
        text=full_text,
        text_json=json.dumps(lines),
        confidence=round(avg_confidence, 4),
    )
