import os

BASE_DIR=os.path.dirname(os.path.abspath(__file__))

DATASET_DIR=os.path.join(BASE_DIR,"dataset")
ENCODINGS_DIR=os.path.join(BASE_DIR,"encodings")
ATTENDANCE_DIR=os.path.join(BASE_DIR,"attendance")
DB_PATH=os.path.join(BASE_DIR,"database.db")

os.makedirs(DATASET_DIR,exist_ok=True)
os.makedirs(ENCODINGS_DIR,exist_ok=True)
os.makedirs(ATTENDANCE_DIR,exist_ok=True)