import numpy as np
import cv2 as cv
import os
import gdown
import uuid
import csv
from common import ROOT_FOLDER


def record(args):
    # Checking if the folder already exists
    if args.folder is None:
        print(
            "Please specify folder for data to be recorded into by adding --folder [name]"
        )
        exit()

    # Create the missing folder and joining it to the objects-ROOT_FOLDER
    target_folder = os.path.join(ROOT_FOLDER, args.folder)
    os.makedirs(target_folder, exist_ok=True)

    cascade_file_path = os.path.join(
        cv.data.haarcascades, "haarcascade_frontalface_default.xml"
    )
    if not os.path.exists(cascade_file_path):
        # Download the cascade file using gdown and a google drive link
        cascade_url = "https://drive.google.com/uc?id=YOUR_CASCADE_FILE_ID"
        output_path = os.path.join(
            cv.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        gdown.download(cascade_url, output_path, quiet=False)

    face_classifier = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    video_capture = cv.VideoCapture(0)
    # 0 is for the default camera, parameter can be changed ayntime and depends on amount of cameras you want to use

    def detect_bounding_box(vid):
        gray_image = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            cv.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        return faces

    frame_count = 0
    save_blocked = False

    while True:
        result, video_frame = video_capture.read()
        # to save raw image
        new_vid = video_frame.copy()
        if result is False:
            print("An error occurred while reading the frame.")
            break
            # terminate the loop if the frame is not read successfully

        # Apply face detection
        faces = detect_bounding_box(video_frame)

        # Display the frame with bounding boxes
        cv.imshow("our face detection project :)", video_frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

        if len(faces) == 1 and not save_blocked:
            # Save the image and create a uuid
            image_filename = os.path.join(target_folder, f"{uuid.uuid4()}.jpg")
            cv.imwrite(image_filename, new_vid)

            # Write face position to CSV file
            csv_file_path = os.path.splitext(image_filename)[0] + ".csv"
            with open(csv_file_path, mode="w", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                for face in faces:
                    x, y, w, h = face
                    csv_writer.writerow([x, y, w, h])

            # Block saving for 30 consecutive frames
            save_blocked = True
            frame_count = 0
        elif save_blocked:
            frame_count += 1
            if frame_count >= 30:
                save_blocked = False

    video_capture.release()
    cv.destroyAllWindows()
