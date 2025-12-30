import cv2
import os
import pickle
import numpy as np
from deepface import DeepFace
from database import connect_db
from config import ENCODINGS_DIR
from datetime import datetime

# ------------------------------
# Load embeddings
# ------------------------------
# this function is use for load saved embeddings from encodings directory.
def load_embeddings():
    path = os.path.join(ENCODINGS_DIR, "embeddings.pkl")
    if not os.path.exists(path):
        print("[WARNING] no embeddings are found.\nplease run encode_faces.py first.")
        return {}

    with open(path, "rb") as f:
        raw = pickle.load(f)

    # convert to numpy arrays and normalize
    embeddings = {}
    for k, v in raw.items():
        arr = np.array(v, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        embeddings[k] = arr

    return embeddings

# ------------------------------
# Distance (math)
# ------------------------------
# this function is use for calculate distance between two embeddings.
def calculate_distance(emb1, emb2):
    a = np.array(emb1, dtype=np.float32)
    b = np.array(emb2, dtype=np.float32)
    return np.linalg.norm(a - b)

# ------------------------------
# Recognition logic
# ------------------------------
# this function is use recognise face by compare live embeddings and stored embeddings
def recognize_face(live_embedding, embeddings_data, threshold=0.6):
    best_match = None
    smallest = float('inf')

    for enrollment, saved_emb in embeddings_data.items():
        # saved_emb is already a normalized numpy array from load_embeddings()
        d = np.linalg.norm(live_embedding - saved_emb)
        if d < smallest:
            smallest = d
            best_match = enrollment

    # Return a tuple (match, distance). Caller decides thresholding.
    if smallest < threshold:
        return best_match, smallest
    else:
        return None, smallest

# ------------------------------
# DB helpers
# ------------------------------
# check if attendance already marked for today (helper uses today's date)
def attendance_exists(enrollment):
    today = datetime.now().strftime("%Y-%m-%d")

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM attendance
        WHERE enrollment = ? AND date = ?
    """, (enrollment, today))

    result = cursor.fetchone()
    conn.close()

    return result is not None

# mark attendance row
def mark_attendance(enrollment, name, status="Present"):
    conn = connect_db()
    cursor = conn.cursor()

    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    cursor.execute("""
        INSERT INTO attendance (enrollment, name, date, time, status)
        VALUES (?, ?, ?, ?, ?)
    """, (enrollment, name, today, time_now, status))

    conn.commit()
    conn.close()

# ------------------------------
# Main: webcam -> embed -> recognize -> mark
# ------------------------------
def start_attendance(threshold=0.7, camera_index=0):
    embeddings = load_embeddings()
    if not embeddings:
        print("[ERROR] No embeddings found. Please run encode_faces.py.")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] cannot open camera.")
        return

    print("[INFO] Attendance started, press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] unable to read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # draw rectangle on face.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # crop face from image.
            if w <= 0 or h <= 0:
                continue
            face_img = frame[y:y+h, x:x+w]

            try:
                # DeepFace expects RGB image (we gave rgb_face)
                # convert BGR (from OpenCV) to RGB for DeepFace
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                rep = DeepFace.represent(
                    img_path=rgb_face,
                    model_name="Facenet512",
                    enforce_detection=False
                )

                # Extract embedding reliably for common DeepFace return formats
                if isinstance(rep, list) and len(rep) > 0 and isinstance(rep[0], dict) and "embedding" in rep[0]:
                    live_embedding = rep[0]["embedding"]
                else:
                    live_embedding = rep  # fallback if DeepFace returned raw array/list

                # Convert to numpy and normalize (VERY IMPORTANT)
                live_embedding = np.array(live_embedding, dtype=np.float32)
                norm = np.linalg.norm(live_embedding)
                if norm > 0:
                    live_embedding = live_embedding / norm

            except Exception as e:
                print(f"[WARN] Embedding failed: {e}")
                continue

            # recognition
            enrollment, distance = recognize_face(live_embedding, embeddings, threshold=threshold)

            # debug: show distance for best match (helps tuning threshold)
            if distance is not None:
                print(f"[DEBUG] Best match distance: {distance:.4f}")

            # default label and color (unknown)
            label = "Unknown Person"
            color = (0, 0, 255)  # red

            if enrollment is not None:
                # fetch name from DB
                conn = connect_db()
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM students WHERE enrollment = ?", (enrollment,))
                row = cursor.fetchone()
                conn.close()

                if row:
                    name = row[0]
                    # check today's attendance (attendance_exists uses today)
                    if attendance_exists(enrollment):
                        label = f"{name} - Already Marked"
                        color = (0, 255, 255)  # yellow
                    else:
                        mark_attendance(enrollment, name)
                        label = f"{name} - Present"
                        color = (0, 255, 0)  # green
                else:
                    label = "Student Not Found"
                    color = (255, 0, 0)  # blue-ish

            # overlay label above rectangle
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # show frame and handle quit
        cv2.imshow("Attendance", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Attendance stopped.")

# run if executed directly
if __name__ == "__main__":
    start_attendance()
