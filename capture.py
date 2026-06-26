"""
capture.py

Dataset Capture Module

Mengambil dataset untuk:

- Number (0-10)
- Alphabet (A-Z)
- Word (bisa, contoh, malu, mau, untuk)

Static Dataset
    → ROI Hand Image

Dynamic Dataset
    → Sequence ROI Hand Image
"""

from pathlib import Path
import time

import cv2
import mediapipe as mp

from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    DATASET_DIR,
    NUMBER_CLASSES,
    ALPHABET_CLASSES,
    WORD_CLASSES,
    COUNTDOWN_SECONDS,
    CAPTURE_INTERVAL,
    SEQUENCE_LENGTH,
    SEQUENCE_INTERVAL,
    IMAGE_SIZE
)


class CaptureApp:

    def __init__(self):

        # ==================================================
        # Dataset
        # ==================================================

        self.dataset_root = DATASET_DIR

        self.modes = {
            "numbers": NUMBER_CLASSES,
            "alphabets": ALPHABET_CLASSES,
            "words": WORD_CLASSES
        }

        self.mode_names = list(self.modes.keys())

        self.mode_index = 0
        self.label_index = 0

        # ==================================================
        # Camera
        # ==================================================

        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                "Webcam tidak dapat dibuka."
            )

        # ==================================================
        # MediaPipe Hands
        # ==================================================

        self.mp_hands = mp.solutions.hands

        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # ==================================================
        # Capture Configuration
        # ==================================================

        self.countdown_seconds = COUNTDOWN_SECONDS

        self.capture_interval = CAPTURE_INTERVAL

        self.sequence_length = SEQUENCE_LENGTH

        self.sequence_interval = SEQUENCE_INTERVAL

        self.image_size = IMAGE_SIZE

        # ==================================================
        # State
        # ==================================================

        self.auto_capture = False

        self.last_capture = 0

        # ==================================================
        # Initialize
        # ==================================================

        self.create_dataset_folder()

    # ======================================================
    # Properties
    # ======================================================

    @property
    def current_mode(self):

        return self.mode_names[self.mode_index]

    @property
    def current_label(self):

        return self.modes[
            self.current_mode
        ][self.label_index]

    @property
    def current_folder(self):

        return (
            self.dataset_root
            / self.current_mode
            / self.current_label
        )

    # ======================================================
    # Dataset
    # ======================================================

    def create_dataset_folder(self):

        for mode, labels in self.modes.items():

            for label in labels:

                (
                    self.dataset_root
                    / mode
                    / label
                ).mkdir(
                    parents=True,
                    exist_ok=True
                )

    def image_count(self):

        if self.current_mode == "words":

            return len([
                folder
                for folder in self.current_folder.iterdir()
                if folder.is_dir()
            ])

        return len(
            list(
                self.current_folder.glob("*.jpg")
            )
        )

    def next_filename(self):

        return (
            f"{self.current_label}_"
            f"{self.image_count()+1:04d}.jpg"
        )

    def next_sequence_folder(self):

        folder = (
            self.current_folder
            / f"seq_{self.image_count()+1:04d}"
        )

        folder.mkdir(
            exist_ok=True
        )

        return folder

    # ======================================================
    # Navigation
    # ======================================================

    def next_mode(self):

        self.mode_index = (
            self.mode_index + 1
        ) % len(self.mode_names)

        self.label_index = 0

        self.auto_capture = False

    def next_label(self):

        self.label_index = (
            self.label_index + 1
        ) % len(
            self.modes[self.current_mode]
        )

        self.auto_capture = False

    def previous_label(self):

        self.label_index = (
            self.label_index - 1
        ) % len(
            self.modes[self.current_mode]
        )

        self.auto_capture = False

        # ======================================================
    # Camera
    # ======================================================

    def read_frame(self):
        """
        Membaca frame webcam.
        """

        success, frame = self.cap.read()

        if not success:
            return None

        return cv2.flip(frame, 1)

    # ======================================================
    # Hand Detection
    # ======================================================

    def process_frame(self, frame):
        """
        Deteksi tangan, gambar landmark,
        lalu crop ROI tangan.
        """

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        original = frame.copy()

        results = self.hands.process(rgb)

        roi = None
        detected = False

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks

            h, w, _ = frame.shape

            xs = []
            ys = []

            for hand in hand:

                self.mp_draw.draw_landmarks(
                    frame,
                    hand,
                    self.mp_hands.HAND_CONNECTIONS
                )

                for lm in hand.landmark:

                    xs.append(int(lm.x * w))
                    ys.append(int(lm.y * h))

            margin = 30

            x1 = max(0, min(xs) - margin)
            y1 = max(0, min(ys) - margin)

            x2 = min(w, max(xs) + margin)
            y2 = min(h, max(ys) + margin)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )

            roi = original[
                y1:y2,
                x1:x2
            ]

            if roi.size > 0:

                roi = cv2.resize(
                    roi,
                    self.image_size
                )

                detected = True

        return frame, roi, detected

    # ======================================================
    # UI
    # ======================================================

    def draw_text(
        self,
        frame,
        text,
        x,
        y,
        color=(0,255,0),
        scale=0.65,
        thickness=2
    ):

        cv2.putText(
            frame,
            text,
            (x,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness
        )

    def draw_information(self, frame):

        status = (
            "RUNNING"
            if self.auto_capture
            else "PAUSED"
        )

        info = [

            f"Mode : {self.current_mode}",

            f"Label : {self.current_label}",

            f"Data : {self.image_count()}",

            f"Status : {status}",

            "",

            "[SPACE] Start",

            "[P] Pause",

            "[TAB] Next Label",

            "[B] Previous Label",

            "[M] Change Mode",

            "[Q] Quit"

        ]

        y = 30

        for text in info:

            self.draw_text(
                frame,
                text,
                20,
                y
            )

            y += 30

        return frame

    def show_preview(
        self,
        frame,
        roi=None
    ):

        frame = self.draw_information(
            frame
        )

        cv2.imshow(
            "Webcam",
            frame
        )

        if roi is not None:

            cv2.imshow(
                "Hand ROI",
                roi
            )
    
        # ======================================================
    # Keyboard
    # ======================================================

    def keyboard_handler(self, key):
        """
        Menangani input keyboard.
        """

        # Quit
        if key == ord("q"):
            return False

        # Start Capture
        elif key == 32:      # SPACE
            self.start_capture()

        # Pause Auto Capture
        elif key == ord("p"):
            self.pause_capture()

        # Next Label
        elif key == 9:       # TAB
            self.next_label()

        # Previous Label
        elif key == ord("b"):
            self.previous_label()

        # Change Mode
        elif key == ord("m"):
            self.next_mode()

        return True
    
    # ======================================================
    # Static Capture
    # ======================================================

    def countdown(self):
        """
        Countdown sebelum capture.
        """

        for second in range(
            self.countdown_seconds,
            0,
            -1
        ):

            frame = self.read_frame()

            if frame is None:
                return False

            frame, roi, detected = self.process_frame(
                frame
            )

            frame = self.draw_information(
                frame
            )

            self.draw_text(
                frame,
                str(second),
                560,
                360,
                color=(0, 0, 255),
                scale=5,
                thickness=8
            )

            self.show_preview(
                frame,
                roi
            )

            cv2.waitKey(1)

            time.sleep(1)

        return True

    # ======================================================

    def start_auto_capture(self):

        if self.auto_capture:
            return

        if not self.countdown():
            return

        self.auto_capture = True

        self.last_capture = time.time()

    # ======================================================

    def pause_capture(self):

        self.auto_capture = False

    # ======================================================

    def save_static_frame(
        self,
        roi
    ):
        """
        Simpan ROI untuk dataset statis.
        """

        if roi is None:
            return

        filename = self.next_filename()

        filepath = (
            self.current_folder
            / filename
        )

        cv2.imwrite(
            str(filepath),
            roi
        )

    # ======================================================

    def update_auto_capture(
        self,
        roi,
        detected
    ):
        """
        Dipanggil setiap frame.
        """

        if not self.auto_capture:
            return

        if not detected:
            return

        current = time.time()

        if (
            current - self.last_capture
            < self.capture_interval
        ):
            return

        self.save_static_frame(
            roi
        )

        self.last_capture = current

        # ======================================================
    # Dynamic Capture
    # ======================================================

    def capture_sequence(self):
        """
        Mengambil satu sequence ROI tangan
        untuk dataset kata.
        """

        if not self.countdown():
            return

        sequence_folder = self.next_sequence_folder()

        start_time = time.time()

        saved = 0

        while saved < self.sequence_length:

            frame = self.read_frame()

            if frame is None:
                break

            frame, roi, detected = self.process_frame(frame)

            frame = self.draw_information(frame)

            if detected:

                filename = (
                    f"frame_{saved + 1:04d}.jpg"
                )

                cv2.imwrite(
                    str(sequence_folder / filename),
                    roi
                )

                saved += 1

            self.draw_text(
                frame,
                f"Recording : {saved}/{self.sequence_length}",
                20,
                360,
                color=(0, 0, 255),
                scale=0.8
            )

            self.show_preview(
                frame,
                roi
            )

            cv2.waitKey(1)

            target = (
                saved
                * self.sequence_interval
            )

            elapsed = (
                time.time()
                - start_time
            )

            if target > elapsed:

                time.sleep(
                    target - elapsed
                )

        frame = self.read_frame()

        if frame is not None:

            frame, roi, _ = self.process_frame(
                frame
            )

            frame = self.draw_information(
                frame
            )

            self.draw_text(
                frame,
                "SEQUENCE SAVED",
                380,
                360,
                color=(0,255,0),
                scale=1,
                thickness=2
            )

            self.show_preview(
                frame,
                roi
            )

            cv2.waitKey(700)

        # ======================================================
    # Capture Controller
    # ======================================================

    def start_capture(self):
        """
        Menentukan proses capture sesuai mode aktif.
        """

        if self.current_mode == "words":

            self.capture_sequence()

        else:

            self.start_auto_capture()

        # ======================================================
    # Main Loop
    # ======================================================

    def run(self):

        print("=" * 50)
        print("      BISINDO DATASET CAPTURE")
        print("=" * 50)

        while True:

            frame = self.read_frame()

            if frame is None:
                break

            frame, roi, detected = self.process_frame(
                frame
            )

            self.update_auto_capture(
                roi,
                detected
            )

            self.show_preview(
                frame,
                roi
            )

            key = cv2.waitKey(1) & 0xFF

            if not self.keyboard_handler(key):
                break

        self.cap.release()

        self.hands.close()

        cv2.destroyAllWindows()

# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    app = CaptureApp()

    app.run()
    
    
