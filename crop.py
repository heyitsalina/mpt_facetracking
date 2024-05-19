import cv2 as cv
from common import ROOT_FOLDER, TRAIN_FOLDER, VAL_FOLDER
import os
import csv
import random

# Quellen
#  - How to iterate over all files/folders in one directory: https://www.tutorialspoint.com/python/os_walk.htm
#  - How to add border to an image: https://www.geeksforgeeks.org/python-opencv-cv2-copymakeborder-method/

# This is the cropping of images


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

            # print(image_files)

            for image in image_files:
                # path to image
                image_path = os.path.join(folder_path, image)

                # read image
                img = cv.imread(image_path)
                # print(img)

                # cv.imshow("Bildf", img)
                # cv.waitKey(0)

                # save csv path
                csv_file_path = os.path.join(
                    folder_path, f"{os.path.splitext(image)[0]}.csv"
                )

                # print(csv_file_path)

                # check if path exists
                if not os.path.exists(csv_file_path):
                    continue

                # open and read csv
                with open(csv_file_path, "r") as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        # read coordiantes
                        x, y, w, h = map(int, row)
                        x2, y2  = x + w, y + h
                        # calculate center from x and y
                        xc = (x + x2) / 2
                        yc = (y + y2) / 2
                        # calculate width
                        wx = abs(x - x2) / 2
                        wy = abs(y - y2) / 2
                        # check if coordinates are valid
                        # if x1 < 0 or y1 < 0 or x2 >= img.shape[1] or y2 >= img.shape[0]:
                        #     print(f"Invalid crop coordinates: ({x1}, {y1}), ({x2}, {y2})")
                        #     continue

                        args.border = float(args.border)
                        # calculate border for x and y
                        border_size_pixels_wx = int((1.0 + args.border) * wx)
                        border_size_pixels_wy = int((1.0 + args.border) * wy)

                        # check if border size is too large
                        # if border_size_pixels >= min(img.shape[:2]):
                        #     print("Border size is too large for the image dimensions")
                        #     continue
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

                        # Bild anzeigen
                        # cv.imshow("Cropped Image", cropped_img)
                        # cv.waitKey(0)
                        
                        # Convert args.split to float
                        split_value = float(args.split)

                        # random split into train and val folder
                        if random.uniform(0.0, 1.0) < split_value:
                            destination_folder = os.path.join(VAL_FOLDER)
                        else:
                            destination_folder = os.path.join(TRAIN_FOLDER)

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


# crop(args)
# create_delete_folders()
# clean_files()

# def crop(args):
# TODO: Crop the full-frame images into individual crops
#   Create the TRAIN_FOLDER and VAL_FOLDER is they are missing (os.mkdir)
#   Clean the folders from all previous files if there are any (os.walk)
#   Iterate over all object folders and for each such folder over all full-frame images
#   Read the image (cv.imread) and the respective file with annotations you have saved earlier (e.g. CSV)
#   Attach the right amount of border to your image (cv.copyMakeBorder)
#   Crop the face with border added and save it to either the TRAIN_FOLDER or VAL_FOLDER
#   You can use
#
#       random.uniform(0.0, 1.0) < float(args.split)
#
#   to decide how to split them.
# if args.border is None:
#     print("Cropping mode requires a border value to be set")
#     exit()

# args.border = float(args.border)
# if args.border < 0 or args.border > 1:
#     print("Border must be between 0 and 1")
#     exit()
