"""
app.py

FastAPI Web Application
for BISINDO Detection System.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from PIL import Image

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect
)

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from config import (
    NUMBER_MODEL,
    ALPHABET_MODEL,
    WORD_MODEL,
    NUMBER_CLASSES,
    ALPHABET_CLASSES,
    WORD_CLASSES,
    SEQUENCE_LENGTH,
)

from predict import (
    StaticPredictor,
    WordPredictor,
)

# ==========================================================
# FastAPI
# ==========================================================

app = FastAPI(
    title="BISINDO Detection System"
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(
    directory="templates"
)

# ==========================================================
# Load Model
# ==========================================================

number_predictor = StaticPredictor(
    model_path=NUMBER_MODEL,
    class_names=NUMBER_CLASSES
)

alphabet_predictor = StaticPredictor(
    model_path=ALPHABET_MODEL,
    class_names=ALPHABET_CLASSES
)

word_predictor = WordPredictor(
    model_path=WORD_MODEL,
    class_names=WORD_CLASSES
)

# ==========================================================
# Utility
# ==========================================================

def decode_image(
    data: bytes
) -> Image.Image:
    """
    Mengubah binary JPEG
    menjadi PIL Image.
    """

    image = np.frombuffer(
        data,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image,
        cv2.IMREAD_COLOR
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(
        image
    )

# ==========================================================
# MediaPipe Hand ROI
# ==========================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def extract_hand_roi(image: Image.Image) -> Image.Image | None:

    frame = np.array(image)

    results = hands.process(frame)

    if not results.multi_hand_landmarks:
        return None

    h, w, _ = frame.shape

    hand = results.multi_hand_landmarks[0]

    xs = [int(lm.x * w) for lm in hand.landmark]
    ys = [int(lm.y * h) for lm in hand.landmark]

    x1 = max(min(xs) - 20, 0)
    y1 = max(min(ys) - 20, 0)
    x2 = min(max(xs) + 20, w)
    y2 = min(max(ys) + 20, h)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    roi = cv2.resize(roi, (224, 224))

    return Image.fromarray(roi)
    

# ==========================================================
# Routes
# ==========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def index(
    request: Request
):

    return templates.TemplateResponse(

        "index.html",

        {

            "request": request

        }

    )

# ==========================================================
# WebSocket
# ==========================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()

    mode = "numbers"

    sequence_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    sequence_buffer.clear()
    

    try:

        while True:

            message = await websocket.receive()

            # ==============================================
            # Binary JPEG Frame
            # ==============================================

            if "bytes" in message:

                data = message["bytes"]

                if data is None:
                    continue

                image = decode_image(data)

                extract_hand_roi()

                image = extract_hand_roi(image)

                if image is None:
                    continue

            # ==============================================
            # Text Command
            # ==============================================

            elif "text" in message:

                command = message["text"]

                if command in (

                    "numbers",

                    "alphabets",

                    "words"

                ):

                    mode = command

                    sequence_buffer.clear()

                continue

            else:

                continue

            # ==============================================
            # Static Model
            # ==============================================

            if mode == "numbers":

                result = number_predictor.predict(
                    image
                )

            elif mode == "alphabets":

                result = alphabet_predictor.predict(
                    image
                )

            # ==============================================
            # Dynamic Model
            # ==============================================

            else:

                sequence_buffer.append(
                    image
                )

                if len(sequence_buffer) < SEQUENCE_LENGTH:

                    await websocket.send_json({

                        "label": "Waiting...",

                        "confidence": 0.0,

                        "index": -1,

                        "progress": len(sequence_buffer),

                        "required": SEQUENCE_LENGTH

                    })

                    continue

                result = word_predictor.predict(

                    list(sequence_buffer)

                )

            # ==============================================
            # Send Result
            # ==============================================

            await websocket.send_json(

                result

            )

    except WebSocketDisconnect:

        sequence_buffer.clear()

        print(

            "Client disconnected."

        )
# ==========================================================
# Events
# ==========================================================

@app.on_event("startup")
async def startup_event() -> None:

    print("=" * 60)
    print("BISINDO Detection System")
    print("Backend Started")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event() -> None:

    print("=" * 60)
    print("Backend Stopped")
    print("=" * 60)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )