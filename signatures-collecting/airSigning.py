# Import opencv for computer vision stuff
import cv2
import numpy as np
# Import hand Tracking Module
import handDetector as htm
import time
import datetime
import sys
import os
from pynput import keyboard

class AirSigning:

    def __init__(self, defaultCam=0):
        self.primaryCam = defaultCam
        # rozmiar okna z kamerą
        self.camHeight, self.camWidth = 1280, 720

        # ilość wykrywanych rąk
        self.detector = htm.handDetector(maxHands=1)

        # kolor rysowania
        self.drawColor = (200, 100, 100) # (141, 43, 193)
        
        # grubość rysowanej kreski
        self.brushThickness = 3
        
        # jak szybko będzie podążać rysowanie za palcem
        self.smooth = 4


        #self.retrySign = 100 #0

    def removeBlackBackground(self, imgCanvas):
        # convert the image to grayscale
        gray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)

        # threshold the image to create a mask
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        # invert the mask
        mask_inv = cv2.bitwise_not(mask)

        # apply the mask to the image
        img_masked = cv2.bitwise_and(imgCanvas, imgCanvas, mask=mask)

        # add an alpha channel to the image
        alpha = np.ones(imgCanvas.shape[:2], dtype=np.uint8) * 255
        alpha[mask_inv == 255] = 0

        return cv2.merge((img_masked, alpha))


    def drawSign(self):
        directory = f"subject{int(input('Enter directory number: '))}"

        parent_dir = "/home/ubuntu/Inzynierka/Over-The-Air-Signatures-Authentication/signatures-database"

        path = os.path.join(parent_dir, directory)

        print(path)

        if not os.path.exists(path):
            os.mkdir(path)
            print(f"Directory '{directory}' created at {path}")
        else:
            print(f"Directory '{directory}' already exists at {path}")
        
        # Connect to webcam
        #cap = cv2.VideoCapture(self.primaryCam)
        # Ustaw rozdzielczość kamery

            #cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # access webcam
        cap = cv2.VideoCapture(0)
        #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        print(cap.get(cv2.CAP_PROP_FPS))
        print(int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, byteorder=sys.byteorder).decode())

        # Sprawdź, czy ustawienia zostały poprawnie zastosowane
        self.camWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'Ustawiona rozdzielczość: {self.camWidth}x{self.camHeight}')

        # Sign Area rectangle
        rectIniWid, rectIniHei = int(self.camWidth * 0.1), int(self.camHeight * 0.1) #int(self.camWidth * 0.1), int(self.camHeight * 0.1)
        rectEndWid, rectEndHei = int(self.camWidth * 0.9), int(self.camHeight * 0.4)
        xPrevious, yPrevious = 0, 0
        imgCanvas = np.zeros((self.camHeight, self.camWidth, 3), np.uint8)
        number = 1
        save_list = []
        while cap.isOpened():
            
            now = datetime.datetime.now()
            current_time = now.strftime("%M:%S:%f")

            ret, frame = cap.read()

            frame = cv2.flip(frame, 1)
            frame = self.detector.findHands(frame, draw=False)
            lmList = self.detector.findPosition(frame, draw=True)

            if len(lmList) != 0:
                # tip of index finger
                indFx, indFy = lmList[8][1:]

                # fingers up detection
                fingers = self.detector.fingerUp()

                # Pause Mode
                if fingers[1] and fingers[2]:
                    xPrevious, yPrevious = 0, 0

                    save_list.append("id: -1, x: -1, y: -1, time: -1")

                    print(f"id: -1, x: -1, y: -1, time: -1")

                # Draw Mode
                if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4] and \
                        rectIniWid < indFx < rectEndWid and rectIniHei < indFy < rectEndHei:

                    cv2.circle(frame, (indFx, indFy), 10, self.drawColor, cv2.FILLED)

                    print(indFx, indFy)

                    #print(f"id: {lmList[8][0]}, x: {lmList[8][1]}, y: {lmList[8][2]}, time: {current_time}", file=save_file)
                    save_list.append(f"id: {lmList[8][0]}, x: {indFx}, y: {indFy}, time: {current_time}")
                    print(f"id: {lmList[8][0]}, x: {indFx}, y: {indFy}, time: {current_time}")

                    if xPrevious == 0 and yPrevious == 0:
                        xPrevious, yPrevious = indFx, indFy

                    indFx = xPrevious + (indFx - xPrevious) // self.smooth
                    indFy = yPrevious + (indFy - yPrevious) // self.smooth

                    cv2.line(imgCanvas, (xPrevious, yPrevious), (indFx, indFy), self.drawColor, self.brushThickness)
                    xPrevious, yPrevious = indFx, indFy

            #    imgCanvas = np.zeros((self.camHeight, self.camWidth, 3), np.uint8)

                #if fingers[1:4] == [1, 1, 1]:
                #    imgCanvas = np.zeros((self.camHeight, self.camWidth, 3), np.uint8)

            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, imgInv)
            frame = cv2.bitwise_or(frame, imgCanvas)

            frame = cv2.rectangle(frame, (rectIniWid, rectIniHei), (rectEndWid, rectEndHei), (0, 78, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            position = (self.camWidth // 2, self.camHeight - 50)
            color = (200, 100, 100)
            cv2.putText(frame, f"sign #{number} | q - quit, s - save, e - erase", position, font, font_scale, color, 2, cv2.LINE_AA)

            cv2.imshow('Webcam', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break


            if key == ord('s'):
                pngOfSign = self.removeBlackBackground(imgCanvas)
                
                file_name = "sign"

                with open(f'/{path}/{file_name}_{number}.txt', 'w') as f:
                    for line in save_list:
                        f.write(f"{line}\n")
                cv2.imwrite(f'/{path}/{file_name}_{number}.png',
                            pngOfSign[rectIniHei:rectEndHei, rectIniWid:rectEndWid])
                number += 1

                imgCanvas = np.zeros((self.camHeight, self.camWidth, 3), np.uint8)
                
                save_list = []

            if key == ord('e'):
                imgCanvas = np.zeros((self.camHeight, self.camWidth, 3), np.uint8)
                save_list = []


        #cap.release()
        #cv2.destroyAllWindows()

        #self.retrySign += 1

        #return "tempSign.png"
    