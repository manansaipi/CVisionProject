from ultralytics import YOLO
import cv2
import math
import cvzone
import os



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
# video
cap = cv2.VideoCapture("./videos/cars.mp4")
# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = "output_video.mp4"
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
# car mask
mask =  cv2.imread("./mask/mask.png")

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    for r in results: 
        boxes = r.boxes
        for box in boxes:
            
            # Bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)  
            # print(x1,y1,x2,y2)

            # take the id of all Classname
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            # Confidance
            conf = math.ceil((box.conf[0]*100))/100
            
            if currentClass == "car" or currentClass == "bus" or currentClass == "truck" \
                or currentClass == "motorbike" and conf > 0.3:
                
                # draw the box
                cv2.rectangle(img, (x1,y1), (x2,y2), classColors[cls], 3)


                # put the class name and the confidance value on the top of the box 
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0,x1), max(40,y1)), colorR=classColors[cls], scale=0.6, thickness=1, offset=3)

    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the VideoWriter and capture objects
out.release()
cap.release()
cv2.destroyAllWindows()
