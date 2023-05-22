import cv2
import os
import random
import shutil

def create_folder(name): 
    for path in os.listdir('image'):
        path_1 = os.path.join('image', path)
        path_2 = os.path.join(path_1, name)
        if os.path.isdir(path_2):
            shutil.rmtree(path_2)
        os.mkdir(path_2)
        if path == 'train':
            path_train = path_2 + "/"
        else:
            path_test = path_2 + "/"
    return(path_train, path_test)

def randomNumber():
    numbers = []
    while len(numbers) < 12:
        num = random.randint(1, 72)
        if num not in numbers:
            numbers.append(num)
    return numbers

def generate_dataset(name):
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    def face_cropped(img):
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(img,1.3,5)
        #scaling factor = 1.3
        #Minimum neighbor
        
        if faces is(): #empty
            return None
        for(x,y,w,h) in faces:
            # cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
            cropped_face = img[y:y+h, x:x+w]
        return cropped_face
    cap = cv2.VideoCapture(0)
    # name = "ADMIN1"
    path_train, path_test = create_folder(name)
    img_id = 0
    id = randomNumber()
    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id += 1
            face = cv2.resize(face_cropped(frame), (200,200))
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            if img_id in id:
                file_name_path = path_test + str(img_id) + ".jpg"
            else:
                file_name_path = path_train + str(img_id) + ".jpg"
                
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            
            cv2.imshow("Cropped face", face)
            if cv2.waitKey(1)==13 or int(img_id) == 72:
                break
    cap.release()
    cv2.destroyAllWindows()
#     print("Collecting samples is completes")
# generate_dataset('Duy')
