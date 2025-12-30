import cv2
import os
from database import add_student,create_tables
from config import DATASET_DIR

#A function to create a folder to store student images.
def create_student_folder(enrollment):
    folder_path=os.path.join(DATASET_DIR,enrollment)
    os.makedirs(folder_path,exist_ok=True)
    return folder_path

#A function to register a student.
def register_student():
    create_tables()

    print("=== REGISTER NEW STUDENT ===")
    name=input("Enter your full name:").strip()
    enrollment=input("enter your enrollment number:").strip()
    class_name=input("enter your class name:").strip()

    if not name or not enrollment:
        print("[ERROR]⚠️please enter your name and enrollment both.")
        return
    #Create a student folder to store a data.
    save_folder=create_student_folder(enrollment)
    print(f"[INFO] Images will be saved in: {save_folder}")

    #USE OUR MAIN TECHNOLOGY TO CAPTURE IMAGES.
    #load a face ditector
    face_cascade=cv2.CascadeClassifier(
        cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
    )

    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR]⚠️can't open camera.")
        return

    img_count=0
    max_images=30

    print("[INFO] Press 'c' to capture,'q' to quit.")

    #make to capture 30 images.
    while True:
        ret,frame=cap.read()
        if not ret:
            print("[ERROR] Unable to read frame.")
            break

        #convert image in gray
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #face detector
        faces = face_cascade.detectMultiScale(gray,1.3,5)  

        #rectangle on face
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        #show image count.
        cv2.putText(
            frame,
            f"Images Captured: {img_count}/{max_images}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        #display the frame
        cv2.imshow("Register Student",frame)

        key=cv2.waitKey(1) & 0xFF

        #capture image
        if key==ord('c'):
            if len(faces)==0:
                print("[WARNING] no face found.")
                continue
            
            (x,y,w,h)=faces[0]
            face_img= frame[y:y+h,x:x+w]

            img_path = os.path.join(save_folder,f"{enrollment}_{img_count}.jpg")
            cv2.imwrite(img_path, face_img)

            print(f"[saved] Image saved at {img_path}")
            img_count+=1

            if img_count>=max_images:
                print("[INFO] Maximum images captured.")
                break   

        #quit
        elif key==ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

    #save a student details and image folder in database.
    #store only when at least one image is captured.

    if img_count>0:
        add_student(name,enrollment, class_name, save_folder)
        print(f"[SUCCESS] Student '{name}' registered successfully!")
    else:
        print("[INFO] no images captured. Registation cancelled")

if __name__=="__main__":
    register_student()