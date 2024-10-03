import os
import time
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import onnxruntime as ort
import numpy as np
from model import CNN
from torchvision.ops import nms
import cv2
import time

class ImagePredictor:
    def __init__(self, cnn_model_path, onnx_model_path, cuda_available=True):
        self.cuda = cuda_available and torch.cuda.is_available()
        self.cnn_model_path = cnn_model_path
        self.onnx_model_path = onnx_model_path
        self._load_cnn_model()
        self._load_onnx_model()
        self.transform_cnn = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor(), 
        ])
        self.transform_detection = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        self.focal_lengths = []
        self.average_focal_length = None
        self.run_count = 0
        self.max_runs = 100
        self.use_average_focal_length = False

    def _load_cnn_model(self):
        self.model = CNN()
        if not os.path.isfile(self.cnn_model_path):
            raise RuntimeError(f"No checkpoint found at '{self.cnn_model_path}'")
        checkpoint = torch.load(self.cnn_model_path, map_location='cuda' if self.cuda else 'cpu')
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Checkpoint '{self.cnn_model_path}' loaded.")
        self.model = self.model.to("cuda") if self.cuda else self.model
        self.model.eval()
        torch.set_grad_enabled(False)

    def _load_onnx_model(self):
        self.session = ort.InferenceSession(self.onnx_model_path)

    def predict_frame(self, frame, confidence_threshold=0.7, nms_threshold=0.3):
        original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        original_size = original_image.size

        # Transform for the CNN
        if not self.use_average_focal_length:
            image_tensor = self.transform_cnn(original_image).unsqueeze(0)
            image_tensor = image_tensor.to("cuda") if self.cuda else image_tensor

            # CNN Prediction
            model_input = {"img": image_tensor}
            output = self.model(model_input)
            focal_length = output[0][0].item()
            print(focal_length)
            # Track focal lengths for averaging
            self.focal_lengths.append(focal_length)
            self.run_count += 1

            if self.run_count == self.max_runs:
                self.average_focal_length = sum(self.focal_lengths) / len(self.focal_lengths)
                self.use_average_focal_length = True
        else:
            focal_length = self.average_focal_length
          

        # Prepare for detection model
        image_for_detection = self.transform_detection(original_image).unsqueeze(0).numpy()

        # Prepare inputs
        inputs = {self.session.get_inputs()[0].name: image_for_detection}

        # Start the timer
        start_time = time.time()

        outputs = self.session.run(None, inputs)
        detections = outputs[0][0]

        filtered_boxes = []
        scores = []

        scale_x = original_size[0] / 640
        scale_y = original_size[1] / 640

        # Process detections
        for i in range(detections.shape[1]):
            det = detections[:, i]
            x, y, w, h = det[:4]
            conf = det[4]

            if conf > confidence_threshold:
                x1 = (x - w / 2) * scale_x
                y1 = (y - h / 2) * scale_y
                x2 = (x + w / 2) * scale_x
                y2 = (y + h / 2) * scale_y

                filtered_boxes.append([x1, y1, x2, y2])
                scores.append(conf)

        if filtered_boxes:
            boxes_tensor = torch.tensor(filtered_boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)

            keep_indices = nms(boxes_tensor, scores_tensor, nms_threshold)
            final_detections = boxes_tensor[keep_indices].numpy()

            # Draw bounding boxes and text on the original image
            frame = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
            for box in final_detections:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

                sensor_height = 24 
                object_height_px = y2 - y1
                object_height_est_mm = 1900
                image_height_px = 720


                distance_mm = ((focal_length * object_height_est_mm * image_height_px) / (object_height_px * sensor_height))

                #convert to meters

                distance_m = distance_mm / 1000

                text = f"Distance: {distance_m:.2f} m"
                print(text)
                cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame


    def predict_video(self, video_path, output_path, confidence_threshold=0.7, nms_threshold=0.3):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    
        count = 0
        while cap.isOpened():
            count = count + 1
            ret, frame = cap.read()
            if not ret:
                break
                # Start timing

            if count % 2 ==0:
                print("skipping")
                continue
            start_time = time.time()
            frame = self.predict_frame(frame, confidence_threshold, nms_threshold)
            out.write(frame)

            # End timing
            end_time = time.time()
        
            # Print the elapsed time
            print(f"Frame_predicted: {end_time - start_time} seconds -- frame {count}")

    
        cap.release()
        out.release()
        cv2.destroyAllWindows()



def main():
    predictor = ImagePredictor(cnn_model_path="./models/best_model.pth", onnx_model_path="./models/model.onnx")
    predictor.predict_video(video_path="./test_videos/iphone_walking_from_camera.mp4", output_path="./output_video.avi")


if __name__ == '__main__':
    main()
