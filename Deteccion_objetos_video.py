# El vídeo es obra de NickyPe

import cv2

# Cargar el modelo
prototxt = "model/MobileNetSSD_deploy.prototxt.txt"
model = "model/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)


# Definición de las clases
classes = {0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat",
           5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow",
           11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike",
           15: "person", 16: "pottedplant", 17: "sheep", 18: "sofa",
           19: "train", 20: "tvmonitor"}

# Inicializamos la captura de video
cap = cv2.VideoCapture("pajaro.mp4")

while True:
     # Se lee un frame de video desde el objeto cap, que representa la cámara.
     # ret es un valor booleano que indica si la lectura fue exitosa, y frame contiene la imagen capturada.
    ret, frame = cap.read()
    if not ret:
         break


    height, width, _ = frame.shape
    frame_resized = cv2.resize(frame, (300, 300))

    # Creación de un blob a partir del frame redimensionado
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))

    # Realizar detecciones
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
          confidence = detections[0, 0, i, 2]
          if confidence > 0.60:
               class_id = int(detections[0, 0, i, 1]) # Obtención de la confianza de la detección
               label = classes.get(class_id, "Unknown")  # Se obtiene el ID de clase y se busca la etiqueta correspondiente en el diccionario classes.

               # Calcular coordenadas de la caja delimitadora. Se calculan en relación con el tamaño original del frame.
               box = detections[0, 0, i, 3:7] * [width, height, width, height]
               x_start, y_start, x_end, y_end = box.astype("int")

               # Dibujar la caja delimitadora y la etiqueta en la imagen
               cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
               cv2.putText(frame, f"Conf: {confidence:.2f}", (x_start, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 0, 0), 2)
               cv2.putText(frame, label, (x_start, y_start - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    # Mostrar el frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Presionar Esc para salir
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()








