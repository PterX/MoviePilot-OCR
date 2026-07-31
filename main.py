import base64
import binascii
import io
import re
from threading import Lock

import cv2
import ddddocr
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
DARK_PIXEL_THRESHOLD = 64
MIN_COMPONENT_AREA = 3
BORDER_WIDTH = 2

app = FastAPI(title="MoviePilot OCR", version="2.0.0")

# The beta model is materially more accurate on MoviePilot's captcha samples.
ocr = ddddocr.DdddOcr(beta=True, show_ad=False)
ocr_lock = Lock()


class OCRRequest(BaseModel):
    base64_img: str


class OCRResponse(BaseModel):
    result: str


class ImageTooLargeError(ValueError):
    pass


class InvalidImageError(ValueError):
    pass


@app.get("/")
def root():
    return {"message": "MoviePilot OCR API"}


@app.post("/captcha/base64", response_model=OCRResponse)
def captcha_base64(data: OCRRequest):
    try:
        image_bytes = decode_base64_image(data.base64_img)
        result = recognize_captcha(image_bytes)
    except ImageTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return OCRResponse(result=result)


def decode_base64_image(value: str) -> bytes:
    payload = "".join((value or "").split())
    if payload.lower().startswith("data:image/"):
        metadata, separator, payload = payload.partition(",")
        if not separator or ";base64" not in metadata.lower():
            raise InvalidImageError("invalid image data URL")

    if not payload:
        raise InvalidImageError("base64_img must not be empty")

    max_encoded_length = ((MAX_IMAGE_BYTES + 2) // 3) * 4
    if len(payload) > max_encoded_length:
        raise ImageTooLargeError("image exceeds the 5 MiB limit")

    padding = len(payload) % 4
    if padding:
        payload += "=" * (4 - padding)

    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError("base64_img is not valid Base64") from exc

    if not image_bytes:
        raise InvalidImageError("decoded image must not be empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageTooLargeError("image exceeds the 5 MiB limit")
    return image_bytes


def preprocess_captcha(image_bytes: bytes) -> bytes:
    gray = load_grayscale_image(image_bytes)
    binary = np.where(gray <= DARK_PIXEL_THRESHOLD, 0, 255).astype(np.uint8)

    foreground = (binary == 0).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )
    keep_labels = stats[:, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA
    keep_labels[0] = False
    cleaned = np.where(keep_labels[labels], 0, 255).astype(np.uint8)

    border = min(BORDER_WIDTH, cleaned.shape[0] // 2, cleaned.shape[1] // 2)
    if border:
        cleaned[:border, :] = 255
        cleaned[-border:, :] = 255
        cleaned[:, :border] = 255
        cleaned[:, -border:] = 255

    encoded, output = cv2.imencode(".png", cleaned)
    if not encoded:
        raise InvalidImageError("failed to preprocess image")
    return output.tobytes()


def load_grayscale_image(image_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                raise InvalidImageError("image dimensions must be positive")
            if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
                raise ImageTooLargeError("image dimensions exceed 4096 x 4096 pixels")

            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                image = Image.alpha_composite(background, rgba).convert("RGB")
            return np.asarray(image.convert("L"), dtype=np.uint8)
    except ImageTooLargeError:
        raise
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("decoded content is not a supported image") from exc


def recognize_captcha(image_bytes: bytes) -> str:
    processed = preprocess_captcha(image_bytes)
    with ocr_lock:
        prediction = ocr.classification(processed)

    # These captchas use uppercase ASCII letters and digits. Normalizing the
    # model output also resolves visually identical upper/lowercase glyphs.
    return "".join(re.findall(r"[A-Za-z0-9]", prediction)).upper()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9899, reload=False)
