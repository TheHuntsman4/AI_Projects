import cv2
import concurrent.futures
from ultralytics import YOLO
import time

# Video setup
video_path = 'cctvFootage.mp4'
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) 

# Output video setup
fourcc = cv2.VideoWriter_fourcc(*'X264')
output_path = 'output.mp4'
output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Model
model = YOLO('yolov8s.pt')

# Frame processing
prev_frame_time = 0
def process_frame(frame):
  global prev_frame_time
  
  # Make predictions
  results = model.predict(frame)

  # Count persons
  person_count = 0
  for pred in results:
    boxes = pred.boxes
    for box in boxes:
      if box.cls[0] == 0:
        person_count += 1
        
  # Calculate FPS  
  new_frame_time = time.time()
  fps = 1 / (new_frame_time - prev_frame_time)
  prev_frame_time = new_frame_time

  # Annotate frame
  cv2.putText(frame, f'Persons: {person_count}', (10, 25), 
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
              
  return frame

# Main loop
while True:
  ret, frame = cap.read()
  if not ret:
    break
  
  frame = process_frame(frame)
  
  output_video.write(frame)
  
cap.release()
output_video.release()