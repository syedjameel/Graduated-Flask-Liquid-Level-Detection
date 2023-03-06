import cv2
from yolo_segmentation import YOLOSegmentation
import os
import math
from realsense_depth import *


HOME = os.getcwd()

dc = DepthCamera()
model = YOLOSegmentation("best.pt")
while True:

    ret, depth_frame, frame = dc.get_frame()
    bboxes, classes, segmentations, scores = model.detect(frame)
    arclength = []
    if classes is not None:
        for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
            #print("bbox:", bbox, "class_id:", class_id, "seg:", seg, "score:", score)
            (x, y, x2, y2) = bbox
            if class_id == 2:
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
                print("width = ", x2-x, "height = ", y2-y)
                print("adjusted width = ", (x2-x)*0.66, "adjusted height = ", (y2-y)*0.85)
                adjusted_width = (x2-x)*0.66
                adjusted_height = (y2-y)*0.85
                volume = math.pi*((adjusted_width/2)**2)*adjusted_height
                print("volume = ", volume/1000 + 22, "milli liter")
                cv2.putText(frame, "Volume : {}".format(round(volume/1000 + 22, 1)),
                            (int(x - 200), int(y + 40)),
                            cv2.FONT_HERSHEY_PLAIN, 1, (100, 255, 0), 2)

            cv2.polylines(frame, [seg], True, (255, 0, 0), 2)

            #frame = cv2.Canny(frame, 30, 200)


    cv2.imshow("image", frame)
    if cv2.waitKey(1) == 27:
        break

dc.release()
cv2.destroyAllWindows()