import os
import cv2
import dlib

face_detector = dlib.get_frontal_face_detector() # type: ignore

def crop_and_overwrite_faces(folder_path):
    for person in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person)
        if not os.path.isdir(person_path):
            continue
        for file in os.listdir(person_path):
            file_path = os.path.join(person_path, file)
            img = cv2.imread(file_path)
            if img is None:
                print(f"Could not read {file_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)

            if len(faces) == 0:
                print(f"No face found in {file_path}")
                continue

            x, y, w, h = faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()
            face_img = img[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            cv2.imwrite(file_path, face_img)
            print(f"[✓] Cropped and saved {file_path}")

if __name__ == "__main__":
    crop_and_overwrite_faces("Image_Data")
