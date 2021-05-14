import PySimpleGUI as sg
import cv2
import os
import json
import time
import shutil
import numpy as np
from PIL import Image
import test

import threading
import random


class Video(object):

    def __init__(self):

        self.deler = 0
        self.run_face = 0
        self.predict = []  # 误差估计数据

        self.recognizer = None
        self.detector = None

        self.datasets = "./datasets"
        self.face_number = "./face_number"
        self.trainer = "./trainer"

        self.button = None
        self.value = None
        self.font = 'Arial'
        self.layout = [
            [sg.Text('ID'), sg.Input(key='id', size=(10, 1)), sg.Text('Age'), sg.Input(key='age', size=(10, 1))],
            [sg.Button('Import face', font=self.font), sg.Button('Training model', font=self.font),
             sg.Button('Recognize faces', key='first', font=self.font), sg.Button('Clean up the model', font=self.font),
             sg.Button('Clean up all data', font=self.font), ],
            [sg.Output(size=(60, 8), font=self.font)],
            [sg.Button('exit', font=self.font), ]]
        self.window = sg.Window('Face recognition system', font=self.font, layout=self.layout)
    def mainloop(self):
        while True:
            self.button, self.value = self.window.read()
            if self.button in ['exit', None]:
                break
            elif self.button in ['Import face']:
                self.face_datasets()
            elif self.button in ['first']:
                element = self.window.find_element('first')
                text = element.GetText()
                if text in ['Recognize faces', 'Start recognize']:
                    print('Start recognize')
                    element.update(text='Stop recognize')
                else:
                    print('Stop recognize')
                    element.update(text='Start recognize')
                self.run_face = self.run_face * -1 + 1
                if self.run_face:
                    t1 = sg.Thread(target=self.face_recognition)
                    t1.setDaemon(True)
                    t1.start()
            elif self.button in ['Training model']:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.detector = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

                faces, ids = self.getImagesAndLabels('datasets')

                self.recognizer.train(faces, np.array(ids))

                self.recognizer.save('trainer/trainer.yml')
            elif self.button in ['Clean up the model']:
                self.cleaner(self.trainer)
            elif self.button in ['Clean up all data']:
                self.clean_all()
            self.button=None

    def face_datasets(self):

        ID = self.value['id']
        age = self.value['age']
        dict1 = {"ID": ID, "age": age}
        for d in range(0,10):
            if os.path.exists('./face_number/'+str(d)+'.json')==False:
                face_id = d
                with open("./face_number/" + str(face_id) + ".json", "w") as f:
                    f.write(json.dumps(dict1))
                break
            pass
        if os.path.exists('./face_number') == True:
            pass
        else:
            os.makedirs('./face_number')
        if os.path.exists('./datasets') == True:
            pass
        else:
            os.makedirs('./datasets')
        if os.path.exists('./trainer') == True:
            pass
        else:
            os.makedirs('./trainer')

        face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')

        count = 0
        cap = cv2.VideoCapture(0)
        cap.set(3,360)
        cap.set(4,360)


        while True:
            _, image_frame = cap.read()

            gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(gray, 1.1, 5)
            for x, y, w, h in faces:
                print("Importing a face---")
                cv2.rectangle(image_frame, (x, y), (x + w + 20, y + h + 20), (255, 255, 0), 2)

                count += 1
                if count > 100:
                    break

                cv2.imwrite("datasets/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

                cv2.imshow('frame', image_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if count > 100:
                print("PS:It has been !")
                break
            elif time.clock() > 3 and count == 0:
                print("PS:The camera does not see the face, please face the camera "
                      "and keep the light sufficient"
                      "or Enter iD and age")
                os.remove('./face_number/' + os.listdir('./face_number')[-1])
                break

        cap.release()

        cv2.destroyAllWindows()

    def face_recognition(self):

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        recognizer.read('trainer/trainer.yml')

        cascadePath = "haarcascade_frontalface_alt2.xml"

        faceCascade = cv2.CascadeClassifier(cascadePath)

        font = cv2.FONT_HERSHEY_SIMPLEX

        cam = cv2.VideoCapture(0)
        cam.set(3,360)
        cam.set(4,360)

        while True:
            if not self.run_face:
                break

            ret, im = cam.read()

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                if not self.run_face:
                    break

                cv2.rectangle(im, (x, y), (x + w + 20, y + h + 20), (0, 255, 0), 4)
                Id = recognizer.predict(gray[y:y + h, x:x + w])
                self.predict.append(Id)
                # with open("./predict2.txt","a+") as f:
                #     f.write(json.dumps(predict)+",")
                if (Id[0] == 1 and Id[1]<=95):
                    with open("face_number/1.json", "r") as f:
                        Id_dict = json.loads(f.read())
                        Id_content = Id_dict["ID"]
                        age_content = Id_dict["age"]
                # If not exist, then it is Unknown
                elif (Id[0] == 2 and Id[1]<=95):
                    with open("face_number/2.json", "r") as f:
                        Id_dict = json.loads(f.read())
                        Id_content = Id_dict["ID"]
                        age_content = Id_dict["age"]
                elif (Id[0] == 3 and Id[1]<=95):
                    with open("face_number/3.json", "r") as f:
                        Id_dict = json.loads(f.read())
                        Id_content = Id_dict["ID"]
                        age_content = Id_dict["age"]
                elif (Id[0] == 4 and Id[1]<=95):
                    with open("face_number/4.json", "r") as f:
                        Id_dict = json.loads(f.read())
                        Id_content = Id_dict["ID"]
                        age_content = Id_dict["age"]
                elif (Id[0] == 5 and Id[1]<=95):
                    with open("face_number/5.json", "r") as f:
                        Id_dict = json.loads(f.read())
                        Id_content = Id_dict["ID"]
                        age_content = Id_dict["age"]
                elif (Id[0] == 6 and Id[1]<=95):
                    with open("face_number/6.json", "r") as f:
                        Id_dict = json.loads(f.read())
                        Id_content = Id_dict["ID"]
                        age_content = Id_dict["age"]
                elif (Id[0] == 7 and Id[1]<=95):
                    with open("face_number/7.json", "r") as f:
                        Id_dict = json.loads(f.read())
                        Id_content = Id_dict["ID"]
                        age_content = Id_dict["age"]

                else:
                    Id_content = "Unknow"
                    age_content = " "

                cv2.rectangle(im, (x - 22, y - 95), (x + w + 22, y - 22), (0, 255, 0), -1)
                cv2.putText(im, str(Id_content + ' ' + age_content), (x, y - 40), font, 2, (255, 255, 255), 3)

            cv2.imshow('image', im)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()

        cv2.destroyAllWindows()

    def getImagesAndLabels(self, path):

        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

        faceSamples = []

        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L')

            img_numpy = np.array(PIL_img, 'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])

            print("The ID that has been identified", id)

            faces = self.detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])

                ids.append(id)

        print("PS:The training is over.")
        return faceSamples, ids

    def cleaner(self, UPLOAD_PATH):
        if os.path.exists(UPLOAD_PATH):
            for i in os.listdir(UPLOAD_PATH):
                path_file = os.path.join(UPLOAD_PATH, i)
                if os.path.isfile(path_file):
                    os.remove(path_file)
                elif os.path.isdir(path_file):
                    shutil.rmtree(path_file)
                else:
                    pass
            print("OK delete uploads data")
        else:
            print("cant find dir and file")

    def clean_all(self):
        self.cleaner(self.datasets)
        self.cleaner(self.face_number)
        self.cleaner(self.trainer)


if __name__ == '__main__':
    myvideo = Video()
    myvideo.mainloop()
