import cv2
import numpy as np
import matplotlib.pyplot as plt
from yolo_segmentation import YOLOSegmentation
import os
import math
from realsense_depth import *
import time
from camera_calibration.CameraCalibration import get_calibration_params


dc = DepthCamera()
model = YOLOSegmentation("best.pt")


ACTUAL_DIAMETER_OF_FLASK = 41.5 # in mm
ACTUAL_HEIGHT_OF_FLASK = 291 # in mm
measured_height = 0
measured_width = 0
PIXEL_PER_MM_HEIGHT_RATIO = 1
PIXEL_PER_MM_HEIGHT_RATIO_2 = 1
theta = 0
bbox_liquid = [0, 0, 0, 0]
ACTUAL_LIQUID_HEIGHT = 0

# get the calibration params
rms, camera_matrix, dist_coefs, _rvecs, _tvecs = get_calibration_params(
    images_path="camera_calibration/realsense-imgs")

while True:
    ret, depth_frame, img = dc.get_frame()

    # Undistort the img
    img = cv2.undistort(img, camera_matrix, dist_coefs)

    # Load the image  IMG20230224211106 IMG20230224210532
    # img = cv2.imread("graduated-flask/raw-photos1/IMG20230224211106.jpg")

    scale_percent = 100 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    #print(img.shape)

    frame = img.copy()
    bboxes, classes, segmentations, scores = model.detect(frame)
    arclength = []
    if classes is not None:
        for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
            # print("bbox:", bbox, "class_id:", class_id, "seg:", seg, "score:", score)
            (x, y, x2, y2) = bbox
            if class_id == 0:
                #print("Class 0 : ", x, y, x2, y2)
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 1)
                measured_flask_height = y2 - y
                #print(measured_flask_height)
                PIXEL_PER_MM_HEIGHT_RATIO = measured_flask_height/ACTUAL_HEIGHT_OF_FLASK
                print("Apparant Flask height = ", np.round((measured_flask_height/PIXEL_PER_MM_HEIGHT_RATIO)*np.cos(theta), 2), " mm")
            if class_id == 1:
                #print("Class 1 : ", x, y, x2, y2)
                cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 1)

                print("width = ", x2 - x, "height = ", y2 - y)
                #print("adjusted width = ", (x2 - x) * 0.66, "adjusted height = ", (y2 - y) * 0.85)
                #adjusted_width = (x2 - x) * 0.66
                #adjusted_height = (y2 - y) * 0.85
                #volume = math.pi * ((adjusted_width / 2) ** 2) * adjusted_height
                #print("volume = ", volume / 1000 + 22, "milli liter")
                #cv2.putText(frame, "Volume : {} ml".format(round(volume / 1000 + 22, 1)),
                #            (int(x - 200), int(y + 40)),
                #            cv2.FONT_HERSHEY_PLAIN, 1, (100, 255, 0), 2)
                cv2.polylines(frame, [seg], True, (255, 0, 0), 2)

                measured_width = x2 - x
                measured_height = y2 - y

                theta = np.arcsin(measured_height/measured_width)
                print("Theta = ", np.round(theta*(180/np.pi), 2), ' degrees')

                APPARANT_HEIGHT = ACTUAL_DIAMETER_OF_FLASK * (measured_height/measured_width)
                print("Apparant height = ", np.round(APPARANT_HEIGHT, 2), " mm")
                #print(np.sin(theta)*ACTUAL_DIAMETER_OF_FLASK) # another way of getting apparant height
                #PIXEL_PER_MM_HEIGHT_RATIO_2 = measured_height/APPARANT_HEIGHT

                # print(seg.shape)
                # b = np.ones(len(seg))
                # xseg = np.array(seg[:, 0])
                # yseg = np.array(seg[:, 1])
                # xseg1 = xseg - xseg.mean()
                # yseg1 = yseg - yseg.mean()
                # new_seg = np.vstack((xseg1, yseg1)).T
                # print(new_seg)
                # print(new_seg.shape)
                # U, S, V = np.linalg.svd(np.stack((xseg1, yseg1)))
                # tt = np.linspace(0, 2 * np.pi, 1000)
                # circle = np.stack((np.cos(tt), np.sin(tt)))  # unit circle
                # N = len(xseg1)
                # transform = np.sqrt(2 / N) * U.dot(np.diag(S))  # transformation matrix
                # fit = transform.dot(circle) + np.array([[xseg.mean()], [yseg.mean()]])
                #
                # param = np.linalg.pinv(np.array(seg))@b
                # print("Param = ", param)
                # actual_param = np.sqrt(1/param)
                # actual_param[0] = (actual_param[0])*(4/13) #correction due to aspect ratio change when training the model
                # print("Actual param = ", actual_param)
                # APPARANT_DIAMETER_OF_FLASK = (actual_param[1]/actual_param[0])*ACTUAL_DIAMETER_OF_FLASK
                # print("THe apparant diameter of the flask = ", APPARANT_DIAMETER_OF_FLASK, " mm")
                # theta = np.arcsin(actual_param[1]/actual_param[0])
                # print("Theta = ", theta*(180/np.pi))
                # frame = cv2.ellipse(frame, (int(xseg.mean()), int(yseg.mean())), (24, 16),
                #                     0, 0, 360, (0, 0, 255), 1)
                # print(fit.shape)
                # A = (fit.T).copy()
                # print(A.shape)
                # B = np.ones(len(A))
                # par = np.linalg.pinv(np.array(A))@B
                # print(par)
                # plt.scatter(seg[:, 0], seg[:, 1])
                # plt.plot(fit[0, :], fit[1, :], 'r')
                # plt.show()
                #some = np.linalg.lstsq(np.array(seg), b)[0].squeeze()
                #print(some)
            if class_id == 2:
                #print("Class 2 : ", x, y, x2, y2)
                bbox_liquid = [x, y, x2, y2]
                measured_liquid_height = y2 - y
                #print(measured_liquid_height)
                measured_liquid_height = measured_liquid_height - (measured_height/2)
                measured_liquid_height_mm = measured_liquid_height/PIXEL_PER_MM_HEIGHT_RATIO
                ACTUAL_LIQUID_HEIGHT = measured_liquid_height_mm/np.cos(theta)
                print("Actual Liquid height ", np.round(ACTUAL_LIQUID_HEIGHT, 2), " mm")
                #print("Apparant measured height with ratio", np.round((measured_height/PIXEL_PER_MM_HEIGHT_RATIO_2), 2), " mm")
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 1)

    img = frame.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Apply a Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    MILLILITER = (np.pi * ((38.2/2)**2)*ACTUAL_LIQUID_HEIGHT)/1000 + 22
    print("Milliliter = ", np.round(MILLILITER, 2), " ml")
    cv2.putText(img, "Volume : {} ml".format(round(MILLILITER, 2)),
                (int(bbox_liquid[0]+50), int(bbox_liquid[1]+100)),
                cv2.FONT_HERSHEY_PLAIN, 1, (100, 255, 0), 2)


    # Display the result
    cv2.imshow("Result1", edges)
    cv2.imshow("Result2", img)
    if cv2.waitKey(1) == 27:
        break
    #time.sleep(0.5)
    print("dep frame ", depth_frame[359, 364])

dc.release()
cv2.destroyAllWindows()
