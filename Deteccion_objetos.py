import cv2

prototxt = 'model/MobileNetSSD_deploy.prototxt.txt'
model = "model/MobileNetSSD_deploy.caffemodel"

# Carga del modelo
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Class labels
classes = {0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat",
           5: "bottle", 6: "bus", 7: "car", 8: "cat",
           9: "chair", 10: "cow", 11: "diningtable", 12: "dog",
           13: "horse", 14: "motorbike", 15: "person", 16: "pottedplant",
           17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}

# Cargamos la imagen
image = cv2.imread("grupo.jpg")

# Etiquetas de clase del modelo
input_shape = (300, 300)
escala = 0.007843
media = (127.5, 127.5, 127.5)
height, width, _ = image.shape
image_resized = cv2.resize(image, (input_shape))

# Preprocesamiento de imágenes
blob = cv2.dnn.blobFromImage(image_resized, escala, input_shape, media)

# Establecer la entrada de la red neuronal con la imagen preprocesada
net.setInput(blob)

# Obtener las detecciones de objetos
detections = net.forward()

# Iterar sobre todas las detecciones encontradas en la imagen
for detection in detections[0][0]:
    # Obtener la confianza de la detección
    confidence = detection[2]

    if confidence > 0.60:
        label = classes[detection[1]]
        # Calcular las coordenadas del cuadro delimitador del objeto detectado
        box = detection[3:7] * [width, height, width, height]
        x_start, y_start, x_end, y_end = map(int, box)

        # Dibuja un rectángulo alrededor del objeto detectado en la imagen original
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

        # Mostrar la confianza de la detección como un texto en la imagen original
        cv2.putText(image, "Confidence: {:.2f}".format(confidence * 100), (x_start, y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Mostrar la etiqueta del objeto detectado como un texto en la imagen original
        cv2.putText(image, label, (x_start, y_start - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
