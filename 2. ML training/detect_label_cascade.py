import cv2
import shutil


def detect_label_cascade(yyyymmdd, path, folder, file, jpg_path):
    refine_path = 'X:/localUser/TrainingData/Refinement/refine_pass/'+yyyymmdd+'_정제완료'
    face_path = 'X:/localUser/TrainingData/Refinement/refine_face'
    jpg_path = path+'/'+folder+'/'+file
    image = cv2.imread(jpg_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    if len(faces) > 0:
        shutil.move(refine_path+'/'+folder, face_path+'/'+folder+'_f')
