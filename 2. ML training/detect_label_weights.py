import numpy as np
import cv2
import shutil




def detect_label_weights(path, label_class):
    detect_label = set()
    net = cv2.dnn.readNet('yolov4.weights',
                          'yolov4.cfg')
    classes = []
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for file in sorted(os.listdir(path)):
        img = cv2.imread(path+'/'+file)
        img = cv2.resize(img, None, fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_AREA)
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(
            img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
            if label == label_class:  # coco.names의 class 중 탐지할 class 선택
                detect_label.add(file)

    return detect_label


def detect_label_weights_move(path, label_class):
    face_path = 'X:/localUser/TrainingData/Refinement/refine_face'
    for folder in detect_label_weights(path, label_class):
        shutil.move(path+'/'+folder, face_path+'/'+folder+'_f')
