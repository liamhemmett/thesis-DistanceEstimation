# Image and Video Prediction System
## Overview
This Python program is designed to predict object distances and detect objects in video frames using a combination of a Convolutional Neural Network (CNN) and an ONNX-based object detection model. It processes each frame from an input video, detects objects, estimates their distances from the camera, and outputs a new video with bounding boxes and distance annotations.

## Features
Focal Length Estimation: Uses a CNN model to estimate the focal length of the camera based on each frame.
Object Detection: Detects objects using an ONNX model, with bounding box generation and non-maximum suppression (NMS) to filter out redundant boxes.
Distance Estimation: Calculates the distance of detected objects based on the focal length and object dimensions.
Video Processing Pipeline: Processes video frames with frame skipping to optimize runtime performance.

## How It Works
### CNN Focal Length Estimation
The program loads a CNN model (from best_model.pth) to predict the focal length of each video frame.
After processing a predefined number of frames (max_runs), the system averages the focal length predictions and uses the average for subsequent frames to stabilize predictions.
###Object Detection and Distance Estimation
The ONNX object detection model (from model.onnx) is used to detect objects in each frame.
Detected objects' bounding boxes are processed with non-maximum suppression (NMS) to eliminate overlapping boxes.
The distance of each detected object from the camera is estimated using the focal length, object dimensions, and image properties.
### Video Prediction
The program processes a video frame-by-frame:

Frames are passed to the CNN model to estimate the camera's focal length.
Objects are detected and distances are calculated.
Bounding boxes and distance annotations are drawn on the frame.
The processed frames are written to an output video file.
## Usage
### Set up your models:
Place the trained CNN model (best_model.pth) and ONNX object detection model (model.onnx) in the appropriate directory.
### Run the program:
To predict object distances in a video and generate an annotated output video, run the following command:

python predict.py

By default, the program processes the video test_videos/iphone_walking_from_camera.mp4 and outputs the result to output_video.avi.

### Modify paths if necessary:
If you need to process a different video or save the output to another location, adjust the video_path and output_path parameters in the main() function of main.py.

## Configuration
### Parameters
cnn_model_path: Path to the CNN model used for focal length estimation.
onnx_model_path: Path to the ONNX model used for object detection.
confidence_threshold: Minimum confidence score for object detection (default: 0.7).
nms_threshold: IoU threshold for non-maximum suppression (default: 0.3).
max_runs: The number of frames after which to use the average focal length for predictions.
### Frame Skipping
The program skips every other frame by default (count % 2 == 0) to improve performance. You can adjust this in the predict_video() method.
##Outputs
Annotated Video: The output video will have bounding boxes drawn around detected objects, and the estimated distance of the object from the camera will be annotated above each box.
Timing Information: The program prints the time taken to process each frame to the console for performance monitoring.

## Docker Support
A Dockerfile is included at the top level of the project. This Dockerfile contains all necessary dependencies to run the program. You can use Docker to quickly set up the environment by building the Docker image and running the program inside the container.

### To build the Docker image:
Copy code

docker build -t image_prediction .

To run the Docker container:

docker run -it image_prediction

### Output:

Checkpoint './models/best_model.pth' loaded.
Frame_predicted: 0.055 seconds -- frame 3
Frame_predicted: 0.052 seconds -- frame 5
The output video with distance annotations will be saved as output_video.avi.

## Credits
This project utilizes focal length estimation based on prior work:

N. Metzger, MLFocalLengths: Estimating the Focal Length of a Single Image, GitHub Repository, 2023, https://github.com/nandometzger/MLFocalLengths