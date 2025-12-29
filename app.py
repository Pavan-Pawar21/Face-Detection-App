import sys
import cv2
import os
import csv
from datetime import datetime
from deepface import DeepFace

from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout,
    QPushButton, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import QTimer, Qt


# ================= CONFIG =================
ATTENDANCE_FILE = "attendance_deepface.csv"
DB_PATH = "dataset"
DISTANCE_THRESHOLD = 0.40
# =========================================


# Create attendance file if not exists
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

marked_today = set()


class FaceAttendanceApp(QWidget):
    def __init__(self):
        super().__init__()

        # ---------- Window Setup ----------
        self.setWindowTitle("AI Face Recognition Attendance System")
        self.setWindowIcon(QIcon("ico.ico"))
        self.resize(800, 760)

        # Enable minimize, maximize, close buttons
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )

        self.setStyleSheet("background-color: #0f172a;")
        self.detection_active = False

        # ---------- Title ----------
        self.title = QLabel("AI Face Recognition Attendance")
        self.title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.title.setStyleSheet("color: white;")
        self.title.setAlignment(Qt.AlignCenter)

        # ---------- Video ----------
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: black; border-radius: 12px;"
        )

        # ---------- Status ----------
        self.status = QLabel("Status: Click Start to begin detection")
        self.status.setFont(QFont("Segoe UI", 11))
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("color: #22c55e;")

        # ---------- Buttons ----------
        self.start_btn = QPushButton("▶ Start")
        self.stop_btn = QPushButton("■ Stop")

        for btn in (self.start_btn, self.stop_btn):
            btn.setFixedSize(160, 45)
            btn.setFont(QFont("Segoe UI", 11, QFont.Bold))

        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #1d4ed8; }
            QPushButton:disabled {
                background-color: #475569;
                color: #cbd5f5;
            }
        """)

        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #b91c1c; }
            QPushButton:disabled {
                background-color: #475569;
                color: #cbd5f5;
            }
        """)

        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)

        # ---------- Layout ----------
        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addSpacing(15)
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        layout.addSpacing(15)
        layout.addWidget(self.status)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # ---------- Camera ----------
        self.cap = cv2.VideoCapture(0)

        # ---------- Timer ----------
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ---------- Start Detection ----------
    def start_detection(self):
        self.detection_active = True
        self.status.setText("Status: Face detection running...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    # ---------- Stop Detection ----------
    def stop_detection(self):
        self.detection_active = False
        self.status.setText("Status: Detection stopped")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # ---------- Main Loop ----------
    def update_frame(self):
        if not self.detection_active:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(frame, (640, 480))
        label_text = "Scanning..."

        try:
            results = DeepFace.find(
                img_path=frame,
                db_path=DB_PATH,
                model_name="Facenet",
                enforce_detection=False
            )

            if len(results) > 0 and not results[0].empty:
                best_match = results[0].iloc[0]
                distance = best_match["distance"]

                if distance < DISTANCE_THRESHOLD:
                    identity = best_match["identity"]
                    name = os.path.basename(os.path.dirname(identity))
                    label_text = f"Recognized: {name}"

                    today = datetime.now().date()
                    if (name, today) not in marked_today:
                        marked_today.add((name, today))
                        with open(ATTENDANCE_FILE, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                name,
                                today,
                                datetime.now().strftime("%H:%M:%S")
                            ])
                        self.status.setText(f"Attendance marked for {name}")
                else:
                    label_text = "Unknown Face"

        except Exception:
            label_text = "Detection Error"

        cv2.putText(
            frame,
            label_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(
            rgb.data, w, h, ch * w, QImage.Format_RGB888
        )
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


# ================= RUN APP =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("ico.ico"))

    window = FaceAttendanceApp()
    window.show()
    sys.exit(app.exec_())
