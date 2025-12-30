import sqlite3
from datetime import datetime
from config import DB_PATH

def connect_db():
    return sqlite3.connect(DB_PATH)

def create_tables():
    conn=connect_db()
    cursor=conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS students(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            enrollment TEXT UNIQUE NOT NULL,
            class_name TEXT,
            folder TEXT,
            registered_on TEXT

        )
        """)

    cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                enrollment TEXT NOT NULL,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """)
    conn.commit()
    conn.close()

def add_student(name:str,enrollment:str,class_name:str,folder:str,):
    conn=connect_db()
    cursor=conn.cursor()    

    cursor.execute("""
        INSERT INTO students(name,enrollment,class_name,folder,registered_on)
        VALUES(?, ?, ?, ?, ?) 
    """,(name,enrollment,class_name,folder,datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_tables()
    print("Database & tables created successfully!")

