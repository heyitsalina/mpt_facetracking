import cv2 as cv
from common import ROOT_FOLDER, TRAIN_FOLDER, VAL_FOLDER
import os
import csv
import random


def create_delete_folders():

    # Create TRAIN_FOLDER if it doesn't exist
    if not os.path.exists(TRAIN_FOLDER):
        os.makedirs(TRAIN_FOLDER)
        print(f"Created folder: {TRAIN_FOLDER}")
    else:
        print(f"TRAIN_FOLDER already exists at: {TRAIN_FOLDER}")

    # Create VAL_FOLDER if it doesn't exist
    if not os.path.exists(VAL_FOLDER):
        os.makedirs(VAL_FOLDER)
        print(f"Created folder: {VAL_FOLDER}")
    else:
        print(f"VAL_FOLDER already exists at: {VAL_FOLDER}")


def clean_files():
    # Function to delete files in TRAIN_FOLDER
    for root, dirs, files in os.walk(TRAIN_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)  # Path to the current file
            os.remove(file_path)  # Remove the file
            print(f"Deleted: {file_path}")  # Output that the file is deleted

            # here same thing different name :)
    for root, dirs, files in os.walk(VAL_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(f"Deleted: {file_path}")


def crop(args):

    for root, dirs, files in os.walk(ROOT_FOLDER):
        for folder_name in dirs:
            # path to images
            folder_path = os.path.join(root, folder_name)

            # lst all images in folder with .jpg
            image_files = [
                file for file in os.listdir(folder_path) if file.endswith(".jpg")
            ]

            for image in image_files:
                # path to image
                image_path = os.path.join(folder_path, image)

                # read image
                img = cv.imread(image_path)

                # save csv path
                csv_file_path = os.path.join(
                    folder_path, f"{os.path.splitext(image)[0]}.csv"
                )

                # check if path exists
                if not os.path.exists(csv_file_path):
                    continue

                # open and read csv
                with open(csv_file_path, "r") as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        # read coordiantes
                        x, y, w, h = map(int, row)
                        x2, y2 = x + w, y + h
                        # calculate center from x and y
                        xc = (x + x2) / 2
                        yc = (y + y2) / 2
                        # calculate width
                        wx = abs(x - x2) / 2
                        wy = abs(y - y2) / 2

                        args.border = float(args.border)
                        # calculate border for x and y
                        border_size_pixels_wx = int((1.0 + args.border) * wx)
                        border_size_pixels_wy = int((1.0 + args.border) * wy)

                        # # expand image with border
                        img_with_border = cv.copyMakeBorder(
                            img,
                            border_size_pixels_wy,
                            border_size_pixels_wy,
                            border_size_pixels_wx,
                            border_size_pixels_wx,
                            cv.BORDER_REFLECT,
                        )

                        # new coordinates with broder
                        x1 = int(xc - (border_size_pixels_wx / 2))
                        y1 = int(yc - (border_size_pixels_wy / 2))
                        x2 = int(xc + (border_size_pixels_wx * 2))
                        y2 = int(yc + (border_size_pixels_wy * 2))

                        # crop image
                        cropped_img = img_with_border[y1:y2, x1:x2]

                        # Convert args.split to float
                        split_value = float(args.split)

                        # random split into train and val folder
                        if random.uniform(0.0, 1.0) < split_value:
                            destination_folder = os.path.join(VAL_FOLDER, folder_name)
                        else:
                            destination_folder = os.path.join(TRAIN_FOLDER, folder_name)

                        # check if destination folder exists
                        if not os.path.exists(destination_folder):
                            os.makedirs(destination_folder)

                        # path for cropped image
                        cropped_image_destination = os.path.join(
                            destination_folder, image
                        )

                        # safe image
                        cv.imwrite(cropped_image_destination, cropped_img)
                        print(
                            f"Face cropped from {image} and saved to {cropped_image_destination}"
                        )
    if args.border is None:
        print("Cropping mode requires a border value to be set")
        exit()

    if args.border < 0 or args.border > 1:
        print("Border must be between 0 and 1")
        exit()
