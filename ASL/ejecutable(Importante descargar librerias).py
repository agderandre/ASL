import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Ruta al modelo .h5, ajustada para funcionar con PyInstaller
model_path = os.path.join(os.path.dirname(__file__), 'ASL.h5')
model = load_model(model_path)

# Definir las etiquetas según tu configuración
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Nada'
]

def extract_features(image):
    image = cv2.resize(image, (64, 64))  # Redimensionar a 64x64
    image = np.expand_dims(image, axis=0)  # Añadir batch dimension
    return image / 255.0  # Normalizar los valores de píxeles

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.rectangle(frame, (50, 50), (350, 350), (0, 255, 0), 2)  
    crop_frame = frame[50:350, 50:350]

    features = extract_features(crop_frame)
    pred = model.predict(features)
    prediction_label = labels[np.argmax(pred)]

    cv2.rectangle(frame, (0, 0), (640, 40), (0, 255, 0), -1)
    accu = "{:.2f}".format(np.max(pred) * 100)
    cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Sign Language Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
