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

    # Initialisiere Webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Lade Haar-Wavelet-Kaskade
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    checkpoint = torch.load('model.pt')
    
        # Lade Klassen-Namen aus dem Checkpoint
    if 'classes' in checkpoint:
        class_names = checkpoint['classes']
    else:
        print("Error: 'classes' not found in checkpoint.")
        return
    
    # Lade Modell und Checkpoint
    nClasses = len(class_names)
    model = Net(nClasses=nClasses)
    model.load_state_dict(checkpoint['model'])
    model.eval()


    # Initialisiere Transformationen
    transform = ValidationTransform

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

            args.border = float(args.border)
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

            x1 = int(xc - border_size_pixels_wx)
            y1 = int(yc - border_size_pixels_wy)
            x2 = int(xc + border_size_pixels_wx)
            y2 = int(yc + border_size_pixels_wy)

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
            name = class_names[label]
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.putText(frame, name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Zeige das Ergebnis
        cv.imshow('Live', frame)

        # Beende bei Tastendruck 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    live()
