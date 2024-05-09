import numpy as np
import cv2 as cv
import os
import gdown
import uuid
import csv
from common import ROOT_FOLDER

# from cascade import create_cascade

# Quellen
#  - How to open the webcam: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
#  - How to run the detector: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
#  - How to download files from google drive: https://github.com/wkentaro/gdown
#  - How to save an image with OpenCV: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html
#  - How to read/write CSV files: https://docs.python.org/3/library/csv.html
#  - How to create new folders: https://www.geeksforgeeks.org/python-os-mkdir-method/

# This is the data recording pipeline
def record(args):
    # TODO:
    #   Implement the recording stage of your pipeline
    #   Create missing folders before you store data in them (os.mkdir)
    #   Open The OpenCV VideoCapture Device to retrieve live images from your webcam (cv.VideoCapture)
    #   Initialize the Haar feature cascade for face recognition from OpenCV (cv.CascadeClassifier)
    #   If the cascade file (haarcascade_frontalface_default.xml) is missing, download it from google drive
    #   Run the cascade on every image to detect possible faces (CascadeClassifier::detectMultiScale)
    #   If there is exactly one face, write the image and the face position to disk in two seperate files (cv.imwrite, csv.writer)
    #   If you have just saved, block saving for 30 consecutive frames to make sure you get good variance of images.
    if args.folder is None:
        print("Please specify folder for data to be recorded into")
        exit()

    face_classifier = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    video_capture = cv.VideoCapture(
        0
    )  # 0 ist die default kamera, parameter kann je nach kameraanzahl geändert werden

    def detect_bounding_box(vid):
        gray_image = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            cv.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        return faces

    while True:

        result, video_frame = video_capture.read()  # read frames from the video

        if result is False:
            print("An error while reading the frame has occurred.")
            break  # terminate the loop if the frame is not read successfully

        # apply the function we created to the video frame
        faces = detect_bounding_box(video_frame)

        cv.imshow(
            "our face detection project :)", video_frame
        )  # display the processed frame in a window named "our face detection project ･ᴗ･"

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv.destroyAllWindows()
