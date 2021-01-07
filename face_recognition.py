import cv2
import os
import numpy as np

subjects = ["", "Biden", "Trump"]

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascPath = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascPath)
    #cascPath = "/usr/share/opencv/lbpcascades/ldbpcascade_frontalface.xml"
    #face_cascade = cv2.CascadeClassifier(cascPath)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    
    if(len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(path):
    faces = []
    labels = []
    dirs = os.listdir(path)
    
    for dir in dirs:
        label = int(dir)
        sub_path = path + "/" + str(label)
        image_names = os.listdir(sub_path)
        
        for image_name in image_names:
            print(image_name)
            image_path = sub_path + "/" + image_name
            
            image = cv2.imread(image_path)

            cv2.imshow("Training image", cv2.resize(image, (400, 500)))
            cv2.waitKey(200)
            
            face, rect = detect_face(image)
            
            if face is not None:
                faces.append(face)
                labels.append(label)
    
    cv2.destroyAllWindows()
    
    return faces, labels

print("Preparing data")
faces, labels = prepare_training_data("/home/pi/training-data/")
print("Data prepared")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))


def predict(test_img):
    print(test_img)
    img = test_img.copy()
    face, rect = detect_face(img)
    label, confidence = face_recognizer.predict(face)
    label_text = subjects[label]
    
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255,0), 2)
    cv2.putText(img, label_text, (rect[0], rect[1]-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
    return img
    
print("Predicting images...")

test_img = cv2.imread("/home/pi/test-data/test1")
predicted_img = predict(test_img)

cv2.imshow("Result", cv2.resize(predicted_img, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()


