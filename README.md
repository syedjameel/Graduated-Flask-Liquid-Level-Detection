# Computer Vision and Deep Learning Enabled Real-Time Liquid Level Detection and Measurement in Transparent Containers

## Problem Statement
Laboratories, particularly those in large oil companies, require a fast and reliable system for detecting and  measuring liquid levels in containers. Current manual methods and some of the pure vision based methods are time-consuming, tedious, and prone to errors leading to delays in product development and increased risk of errors. There is a critical need for a fast and reliable system that can detect and measure liquid levels in real-time and improve the efficiency and reducing human errors in laboratory testing procedures. 

Moreover, with the increase in the trend of Industry 4.0 the capability to detect and measure the liquid levels in the transparent flasks/test tubes is a crucial part in the whole perception system of an autonomous robot and by the use of the Computer vision and deeplearning we can easily help solve this problem. The final developed system can be used in various applications across various industries, including pharmaceutical, manufacturing and chemical processing making it a crucial component in the perception system of an autonomous robot in the Industry 4.0. 

## Proposed Solution
To solve the problem of manual and slow pure vision based liquid level measurement in laboratories [1], we propose to develop an fast and accurate automated system that uses the state-of-the-art deep learning technology [2] to provide accurate and real-time feedback on liquid levels in containers from various viewing angles. The system consists of a realsense d435 camera that collects data [4], which is then analyzed by a state-of-the-art deep learning model and angle correction by devising a novel algorithm to correct the errors in any viewing angle and then to determine the liquid level in real-time. This technology will reduce the risk of errors and increase efficiency in laboratory testing procedures, allowing for faster product development and improved quality control. Additionally, the system can be extended to handle datasets with different types of containers and liquids [3] to ensure its effectiveness across various laboratory testing scenarios. The system will be robot-friendly, with a simple interface that can easily pass the real-time measurement data to the microprocessor to further control a robotic arm to pick up the container which is with appropriate liquid level. Once developed, the system will be tested and optimized to ensure its accuracy and reliability, and can be further enhanced with training heavier deep learning models using different types of containers dataset to improve the accuracy of the liquid level measurement over time.

## Methodology
This section is divided into xx subsections:

### Dataset Creation on Roboflow
For dataset creation we choose the roboflow platform as it have tools which will make the process of the annotation of the images seemless. We took 214 images of the graduated flask with different liquid levels at various orientations and annotated it on Roboflow[3].

### Training Yolo v8 on our dataset
After creating our dataset now it was time to train a large model of  Yolo v8 (yolov8l-seg.pt) with 100 epochs input image size as 640x640 pixels. 
![Screenshot](/yolov8_segmentation/images/confusion-matrix.png)




## Timeline
-TODO

## Technologies
1. Deep Learning Model (e.g. YOLOv8)
2. Computer Vision (e.g. OpenCV)
3. Realsense D435 camera
4. Roboflow

## References
1. Ma, H. and Peng, L., 2019, December. Vision based liquid level detection and bubble area segmentation in liquor distillation. In 2019 IEEE International Conference on Imaging Systems and Techniques (IST) (pp.1-6). IEEE.
2. State-of-the-art Deep learning platform YOLOv8 
3. Tool to create Dataset and Annotations Roboflow
4. Realsense depth camera D435 