import cv2 as cv
from common import ROOT_FOLDER, TRAIN_FOLDER, VAL_FOLDER
import os
import csv
import random

# Quellen
#  - How to iterate over all files/folders in one directory: https://www.tutorialspoint.com/python/os_walk.htm
#  - How to add border to an image: https://www.geeksforgeeks.org/python-opencv-cv2-copymakeborder-method/

# This is the cropping of images

# small
def create_delete_folders():
    train_folder = "TRAIN_FOLDER"
    val_folder = "VAL_FOLDER"

    # Create TRAIN_FOLDER if it doesn't exist
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        print(f"Created folder: {train_folder}")
    else:
        print(f"TRAIN_FOLDER already exists at: {train_folder}")

    # Create VAL_FOLDER if it doesn't exist
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
        print(f"Created folder: {val_folder}")
    else:
        print(f"VAL_FOLDER already exists at: {val_folder}")
        
def clean_files():
    # Function to delete files in TRAIN_FOLDER
    for root, dirs, files in os.walk("TRAIN_FOLDER"):
        for file in files:
            file_path = os.path.join(root, file)  # Path to the current file
            os.remove(file_path)  # Remove the file
            print(f"Deleted: {file_path}")  # Output that the file is deleted
            
            #here same thing different name :)
    for root, dirs, files in os.walk("VAL_FOLDER"):
        for file in files:
            file_path = os.path.join(root, file)  
            os.remove(file_path)  
            print(f"Deleted: {file_path}")  


create_delete_folders()
clean_files()

def crop(args):
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
    if args.border is None:
        print("Cropping mode requires a border value to be set")
        exit()

    args.border = float(args.border)
    if args.border < 0 or args.border > 1:
        print("Border must be between 0 and 1")
        exit()
