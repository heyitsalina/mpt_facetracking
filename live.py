import cv2 as cv
import torch
import os
from network import Net
from transforms import ValidationTransform
from PIL import Image


def live(args):
    if args.border is None:
        print("Live mode requires a border value to be set")
        exit()

    args.border = float(args.border)
    if args.border < 0 or args.border > 1:
        print("Border must be between 0 and 1")
        exit()

    # Initialize the video capture device
    cap = cv.VideoCapture(0)
    # Check if the video capture device is opened
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load the Haar-Wavelet-Kaskade for face detection
    face_cascade = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Load the model and checkpoint
    checkpoint = torch.load("model.pt")

    # Load the classes from the checkpoint
    if "classes" in checkpoint:
        class_names = checkpoint["classes"]
    else:
        print("Error: 'classes' not found in checkpoint.")
        return

    # Load the model
    nClasses = len(class_names)
    model = Net(nClasses=nClasses)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Initialize the transformation
    transform = ValidationTransform

    # Loop through the video frames
    while True:
        # Capture a frame from the video
        ret, frame = cap.read()
        # Check if the frame is captured
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Loop through the detected faces
        for (x, y, w, h) in faces:
            # Calculate the center and width of the face
            x2, y2 = x + w, y + h
            xc = (x + x2) / 2
            yc = (y + y2) / 2
            wx = abs(x - x2) / 2
            wy = abs(y - y2) / 2

            # Calculate the border size
            args.border = float(args.border)
            border_size_pixels_wx = int((1.0 + args.border) * wx)
            border_size_pixels_wy = int((1.0 + args.border) * wy)
            
            # Ensure the border size is within the frame dimensions
            border_size_pixels_wx = min(border_size_pixels_wx, frame.shape[1] // 2)
            border_size_pixels_wy = min(border_size_pixels_wy, frame.shape[0] // 2)

            # Expand the image with the border
            img_with_border = cv.copyMakeBorder(
                frame,
                border_size_pixels_wy,
                border_size_pixels_wy,
                border_size_pixels_wx,
                border_size_pixels_wx,
                cv.BORDER_REFLECT,
            )

            # Calculate the new coordinates with the border
            x1 = int(xc - border_size_pixels_wx)
            y1 = int(yc - border_size_pixels_wy)
            x2 = int(xc + border_size_pixels_wx)
            y2 = int(yc + border_size_pixels_wy)
            
            # Ensure the coordinates are within the img_with_border dimensions
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > img_with_border.shape[1]: x2 = img_with_border.shape[1]
            if y2 > img_with_border.shape[0]: y2 = img_with_border.shape[0]            

            # Crop the face from the image with the border
            cropped_img = img_with_border[y1:y2, x1:x2]

            # Convert the cropped image to a PIL image
            pil_image = Image.fromarray(cv.cvtColor(cropped_img, cv.COLOR_BGR2RGB))

            # Convert the PIL image to a PyTorch tensor
            input_tensor = transform(pil_image).unsqueeze(0)

            # Convert the PIL image to a PyTorch tensor
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                label = predicted.item()

            # Draw a rectangle around the face and display the class name
            name = class_names[label]
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.putText(
                frame, name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2
            )

        # Display the result
        cv.imshow("Live", frame)

        # Check if the 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    # Release the video capture device
    cap.release()

    # Close all OpenCV windows
    cv.destroyAllWindows()


if __name__ == "__main__":
    live()
