from ultralytics import YOLO
import cv2
import math
import cvzone
# import cvzone

# results = model("bus.jpg", show=True)
# cv2.waitKey(0)

# open the webcam 
cap = cv2.VideoCapture(1)

# set width and height
cap.set(3, 1280)
cap.set(4, 720)

# model
model = YOLO("./yolo-weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ] 
classColors = [
    (255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255), (255, 0, 255, 255),
    (0, 255, 255, 255), (255, 127, 0, 255), (127, 255, 0, 255), (0, 255, 127, 255), (0, 255, 255, 255),
    (255, 0, 127, 255), (255, 0, 255, 255), (255, 127, 0, 255), (127, 255, 0, 255), (0, 255, 127, 255),
    (255, 0, 255, 255), (127, 255, 0, 255), (0, 255, 127, 255), (255, 0, 0, 255), (0, 255, 0, 255),
    (0, 0, 255, 255), (255, 255, 0, 255), (255, 0, 255, 255), (0, 255, 255, 255), (255, 127, 0, 255),
    (127, 255, 0, 255), (0, 255, 127, 255), (0, 255, 255, 255), (255, 0, 127, 255), (255, 0, 255, 255),
    (255, 127, 0, 255), (127, 255, 0, 255), (0, 255, 127, 255), (255, 0, 255, 255), (127, 255, 0, 255),
    (0, 255, 127, 255), (255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255),
    (255, 0, 255, 255), (0, 255, 255, 255), (255, 127, 0, 255), (127, 255, 0, 255), (0, 255, 127, 255),
    (0, 255, 255, 255), (255, 0, 127, 255), (255, 0, 255, 255), (255, 127, 0, 255), (127, 255, 0, 255),
    (0, 255, 127, 255), (255, 0, 255, 255), (127, 255, 0, 255), (0, 255, 127, 255), (255, 0, 0, 255),
    (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255), (255, 0, 255, 255), (0, 255, 255, 255),
    (255, 127, 0, 255), (127, 255, 0, 255), (0, 255, 127, 255), (0, 255, 255, 255), (255, 0, 127, 255),
    (255, 0, 255, 255), (255, 127, 0, 255), (127, 255, 0, 255), (0, 255, 127, 255), (255, 0, 255, 255),
    (127, 255, 0, 255), (0, 255, 127, 255), (255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255),
    (255, 255, 0, 255), (255, 0, 255, 255), (0, 255, 255, 255), (255, 127, 0, 255), (127, 255, 0, 255),
    (0, 255, 127, 255), (0, 255, 255, 255), (255, 0, 127, 255), (255, 0, 255, 255), (255, 127, 0, 255),
    (127, 255, 0, 255), (0, 255, 127, 255), (255, 0, 255, 255), (127, 255, 0, 255), (0, 255, 127, 255),
    (255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255), (255, 0, 255, 255),
    (0, 255, 255, 255), (255, 127, 0, 255), (127, 255, 0, 255), (0, 255, 127, 255), (0, 255, 255, 255),
    (255, 0, 127, 255), (255, 0, 255, 255), (255, 127, 0, 255), (127, 255, 0, 255), (0, 255, 127, 255),
    (255, 0, 255, 255), (127, 255, 0, 255), (0, 255, 127, 255), (0, 0, 0, 255), (0, 0, 0, 255)
]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results: 
        boxes = r.boxes
        for box in boxes:
            
            # Bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)  
            print(x1,y1,x2,y2)

            # take the id of all Classname
            cls = int(box.cls[0])
            
            # draw the box
            cv2.rectangle(img, (x1,y1), (x2,y2), classColors[cls], 3)

            # Confidance
            conf = math.ceil((box.conf[0]*100))/100

            # put the class name and the confidance value on the top of the box 
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0,x1), max(40,y1)), colorR=classColors[cls])

    cv2.imshow("Image", img)
    cv2.waitKey(1)























# import sys
# from ultralytics import YOLO
# import cv2
# try:
#     # Load a model
#     # model = YOLO("yolov8n.yaml")  # build a new model from scratch
#     model = YOLO("./yolo-weights/yolov8l.pt")  # load a pretrained model (recommended for training)

#     # Use the model
#     # model.train(data="coco128.yaml", epochs=3)  # train the model
#     # metrics = model.val()  # evaluate model performance on the validation set
#     results = model("bus.jpg", show=True)  # predict on an image
#     # path = model.export(format="onnx")  # export the model to ONNX format
#     cv2.waitKey(0)
# except KeyboardInterrupt:
#     print("Execution interrupted by user.")
#     # Perform cleanup actions here, if needed
#     # ...
#     sys.exit(1)