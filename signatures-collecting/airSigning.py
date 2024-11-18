import cv2
import numpy as np
import os
import datetime


class airSigning:
    def __init__(self, handDetector, defaultCam=0):
        self.primaryCam = defaultCam
        self.camHeight, self.camWidth = 720, 1280
        self.detector = handDetector
        self.drawColor = (200, 100, 100)
        self.brushThickness = 3
        self.smooth = 4

    def removeBlackBackground(self, imgCanvas):
        # Konwersja obrazu do skali szarości
        # Convert the image to grayscale
        gray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)

        # Progowanie obrazu w celu utworzenia maski
        # Thresholding the image to create a mask
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        # Odwrócenie maski
        # Inverting the mask
        mask_inv = cv2.bitwise_not(mask)

        # Zastosowanie maski do obrazu
        # Applying the mask to the image
        img_masked = cv2.bitwise_and(imgCanvas, imgCanvas, mask=mask)

        # Dodanie kanału alfa do obrazu
        # Adding an alpha channel to the image
        alpha = np.ones(imgCanvas.shape[:2], dtype=np.uint8) * 255
        alpha[mask_inv == 255] = 0

        return cv2.merge((img_masked, alpha))

    def drawSign(self):
        # Połączenie z kamerą internetową
        # Connecting to the webcam
        cap = cv2.VideoCapture(self.primaryCam)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camHeight)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camWidth)
        self.camWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

        # Obszar prostokąta dla podpisów
        # Rectangle area for signatures
        rectIniWid, rectIniHei = int(self.camWidth * 0.1), int(self.camHeight * 0.1)
        rectEndWid, rectEndHei = int(self.camWidth * 0.9), int(self.camHeight * 0.4)
        xPrevious, yPrevious = 0, 0
        imgCanvas = np.zeros((self.camHeight, self.camWidth, 3), np.uint8)

        # Tworzenie katalogu dla podmiotów
        # Creating a directory for subjects
        directory = f"subject{int(input('Enter directory number: '))}"
        parent_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "signatures-database",
        )
        path = os.path.join(parent_dir, directory)
        os.makedirs(path, exist_ok=True)

        sign_number = 0
        points_in_time_list = []

        while cap.isOpened():
            ret, frame = cap.read()

            frame = cv2.flip(frame, 1)  # Odbicie lustrzane obrazu w poziomie
                                        # Mirroring the image horizontally
            frame = self.detector.findHands(frame, draw=True)
            lmList = self.detector.findPosition(frame, draw=True)

            current_datetime = datetime.datetime.now().strftime("%M:%S:%f")

            if len(lmList) != 0:
                # Czubek palca wskazującego
                # Tip of the index finger
                indFx, indFy = lmList[8][1:]

                # Wykrywanie palców w górze
                # Detecting fingers up
                fingers = self.detector.fingerUp()

                # Tryb pauzy
                # Pause mode
                if fingers[1] and fingers[2]:
                    xPrevious, yPrevious = 0, 0

                    points_in_time_list.append("id: -1, x: -1, y: -1, time: -1")

                # Tryb pisania
                # Writing mode
                if (
                    fingers[1]
                    and not fingers[2]
                    and not fingers[3]
                    and not fingers[4]
                    and rectIniWid < indFx < rectEndWid
                    and rectIniHei < indFy < rectEndHei
                ):

                    # Wypełniony okrąg jako czubek palca wskazującego
                    # Filled circle as the tip of the index finger
                    cv2.circle(frame, (indFx, indFy), 10, self.drawColor, cv2.FILLED)

                    # Zapis współrzędnych punktów w czasie
                    # Saving the coordinates of points over time
                    points_in_time_list.append(
                        f"id: {lmList[8][0]}, x: {indFx}, y: {indFy}, time: {current_datetime}"
                    )

                    # Rysowanie linii między kolejnymi punktami
                    # Drawing a line between consecutive points
                    if xPrevious == 0 and yPrevious == 0:
                        xPrevious, yPrevious = indFx, indFy

                    indFx = xPrevious + (indFx - xPrevious) // self.smooth
                    indFy = yPrevious + (indFy - yPrevious) // self.smooth

                    cv2.line(
                        imgCanvas,
                        (xPrevious, yPrevious),
                        (indFx, indFy),
                        self.drawColor,
                        self.brushThickness,
                    )
                    xPrevious, yPrevious = indFx, indFy

            # Połączenie ekranu kamery (ramka - ang. frame) z obrazem podpisu (płótno - ang. canvas)
            # Combining the webcam screen (frame) with the signature image (canvas)
            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, imgInv)
            frame = cv2.bitwise_or(frame, imgCanvas)

            # Dodanie prostokąta do ramki
            # Adding a rectangle to the frame
            frame = cv2.rectangle(
                frame, (rectIniWid, rectIniHei), (rectEndWid, rectEndHei), (0, 78, 0), 2
            )

            # Dodanie tekstu instrukcji do ramki
            # Adding instruction text to the frame
            font_type = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            position = (self.camWidth // 2, self.camHeight - 50)
            color = (200, 100, 100)
            cv2.putText(
                frame,
                f"sign #{sign_number} | q - quit, s - save, e - erase",
                position,
                font_type,
                font_scale,
                color,
                2,
                cv2.LINE_AA,
            )

            # Wyświetlenie ramki
            # Displaying the frame
            cv2.imshow("Webcam", frame)

            key = cv2.waitKey(1) & 0xFF

            # Zakończenie programu
            # Ending the program
            if key == ord("q"):
                break

            # Zapis podpisu w pliku tekstowym i graficznym
            # Saving the signature to a text and image file
            if key == ord("s"):
                sign_picture = self.removeBlackBackground(imgCanvas)

                sign_number += 1

                with open(f"{path}/sign_{sign_number}.txt", "w") as f:
                    for point_in_time in points_in_time_list:
                        f.write(f"{point_in_time}\n")

                cv2.imwrite(
                    f"{path}/sign_{sign_number}.png",
                    sign_picture[rectIniHei:rectEndHei, rectIniWid:rectEndWid],
                )

                imgCanvas = np.zeros((self.camHeight, self.camWidth, 3), np.uint8)
                points_in_time_list = []

            # Wyczyszczenie płótna
            # Clearing the canvas
            if key == ord("e"):
                imgCanvas = np.zeros((self.camHeight, self.camWidth, 3), np.uint8)
                points_in_time_list = []

        cap.release()
        cv2.destroyAllWindows()
