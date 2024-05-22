import cv2 as cv
import torch
import os
from network import Net

# from cascade import create_cascade
from transforms import ValidationTransform
from PIL import Image

# NOTE: This will be the live execution of your pipeline


def live(args):
    # TODO:
    #   Load the model checkpoint from a previous training session (check code in train.py)
    #   Initialize the face recognition cascade again (reuse code if possible)
    #   Also, create a video capture device to retrieve live footage from the webcam.
    #   Attach border to the whole video frame for later cropping.
    #   Run the cascade on each image, crop all faces with border.
    #   Run each cropped face through the network to get a class prediction.
    #   Retrieve the predicted persons name from the checkpoint and display it in the image
    face_classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    #load model
    checkpoint = torch.load("model.pt", map_location=torch.device("cpu"))
    
    
    
    if args.border is None:
        print("Live mode requires a border value to be set")
        exit()

    args.border = float(args.border)
    if args.border < 0 or args.border > 1:
        print("Border must be between 0 and 1")
        exit()


def live(args):
    # Initialisiere Webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Lade Haar-Wavelet-Kaskade
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Lade Modell und Checkpoint
    model = Net()
    checkpoint = torch.laod('model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialisiere Transformationen
    transform = ValidationTransform()

    while True:
        # Erfasse Frame
        ret, frame = cap.read()
        if not ret:
            break

        # Konvertiere zu Graustufen
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Erkenne Gesichter
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            x2, y2 = x + w, y + h
            xc = (x + x2) / 2
            yc = (y + y2) / 2
            wx = abs(x - x2) / 2
            wy = abs(y - y2) / 2

            args.border = float(args.border)  # Beispielwert, anpassen falls nötig
            border_size_pixels_wx = int((1.0 + args.border) * wx)
            border_size_pixels_wy = int((1.0 + args.border) * wy)

            img_with_border = cv.copyMakeBorder(
                frame,
                border_size_pixels_wy,
                border_size_pixels_wy,
                border_size_pixels_wx,
                border_size_pixels_wx,
                cv.BORDER_REFLECT,
            )

            x1 = int(xc - (border_size_pixels_wx / 2))
            y1 = int(yc - (border_size_pixels_wy / 2))
            x2 = int(xc + (border_size_pixels_wx * 2))
            y2 = int(yc + (border_size_pixels_wy * 2))

            # Zuschneiden des Bildes
            cropped_img = img_with_border[y1:y2, x1:x2]

            # Wandle Gesichtsausschnitt in PIL-Bild um
            pil_image = Image.fromarray(cv.cvtColor(cropped_img, cv.COLOR_BGR2RGB))

            # Wandle in PyTorch Tensor um
            input_tensor = transform(pil_image).unsqueeze(0)

            # Klassifiziere das Gesicht
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                label = predicted.item()

            # Zeichne Rechteck und Namen
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv.putText(frame, str(label), (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Zeige das Ergebnis
        cv.imshow('Live', frame)

        # Beende bei Tastendruck 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    live()