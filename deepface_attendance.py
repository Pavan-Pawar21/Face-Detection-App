import cv2
from deepface import DeepFace
from datetime import datetime
import csv
import os

attendance_file = "attendance_deepface.csv"

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

marked_today = set()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    try:
        results = DeepFace.find(
            img_path=frame,
            db_path="dataset",
            enforce_detection=False,
            model_name="Facenet"
        )

        if len(results) > 0 and not results[0].empty:
            identity = results[0].iloc[0]["identity"]
            name = identity.split(os.sep)[1]

            today = datetime.now().date()

            if (name, today) not in marked_today:
                marked_today.add((name, today))
                with open(attendance_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        name,
                        today,
                        datetime.now().strftime("%H:%M:%S")
                    ])
                print(f"Attendance marked for {name}")

            cv2.putText(
                frame,
                name,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    except:
        pass

    cv2.imshow("Deep Learning Face Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
