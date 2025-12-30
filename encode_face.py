import os
import pickle
import cv2
import numpy as np
from deepface import DeepFace
from config import DATASET_DIR, ENCODINGS_DIR

def load_images(folder):
    images = []
    for fn in os.listdir(folder):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, fn)
            img = cv2.imread(path)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(rgb)
    return images

def generate_embedding(img):
    """Generate a normalized embedding for the image."""
    rep = DeepFace.represent(
        img_path=img,
        model_name="Facenet512",
        enforce_detection=False
    )

    if isinstance(rep, list) and len(rep) > 0 and "embedding" in rep[0]:
        emb = np.array(rep[0]["embedding"], dtype=np.float32)
    else:
        emb = np.array(rep, dtype=np.float32)

    # Normalize (very important!)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    return emb

def encode_all_students():
    embeddings_data = {}

    if not os.path.exists(DATASET_DIR):
        print("[ERROR] Dataset folder not found.")
        return

    for enrollment in os.listdir(DATASET_DIR):
        folder = os.path.join(DATASET_DIR, enrollment)
        if not os.path.isdir(folder):
            continue

        print(f"[INFO] Encoding student {enrollment}...")

        images = load_images(folder)
        if len(images) == 0:
            print(f"[WARN] No images found for {enrollment}.")
            continue

        emb_list = []
        for img in images:
            try:
                emb = generate_embedding(img)
                emb_list.append(emb)
            except Exception as e:
                print(f"[ERROR] embedding failed for {enrollment}: {e}")

        if len(emb_list) == 0:
            print(f"[WARN] No embeddings generated for {enrollment}.")
            continue

        # Compute mean embedding
        stacked = np.stack(emb_list, axis=0)  # shape (N, 512)
        avg_emb = np.mean(stacked, axis=0)

        # Normalize final average embedding
        norm = np.linalg.norm(avg_emb)
        if norm > 0:
            avg_emb = avg_emb / norm

        embeddings_data[enrollment] = avg_emb.tolist()

        print(f"[SUCCESS] Encoded {enrollment}")

    # Save all embeddings
    os.makedirs(ENCODINGS_DIR, exist_ok=True)
    save_path = os.path.join(ENCODINGS_DIR, "embeddings.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(embeddings_data, f)

    print(f"[INFO] Saved embeddings for {len(embeddings_data)} students.")
    print(f"[INFO] File: {save_path}")

if __name__ == "__main__":
    encode_all_students()