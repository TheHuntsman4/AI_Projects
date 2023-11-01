import cv2
import concurrent.futures
from ultralytics import YOLO
import time
from collections import deque

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
processed_frames = deque(maxlen=1000)

# Batch size
batch_size = 10

def process_batch(frames, frame_indices):
  processed_frames = []
  for frame, frame_index in zip(frames, frame_indices):
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
                
    processed_frames.append((frame, frame_index))
    
  return processed_frames

def save_frame(frame_index):
  while True:
    processed_frame, processed_frame_index = processed_frames[0]
    if processed_frame_index == frame_index:
      output_video.write(processed_frame)
      processed_frames.popleft()
      break
    time.sleep(0.01)

# Main loop
with concurrent.futures.ThreadPoolExecutor() as executor:
  frame_index = 0
  frames = []
  frame_indices = []
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    frames.append(frame)
    frame_indices.append(frame_index)
    
    if len(frames) == batch_size:
      executor.submit(process_batch, frames, frame_indices).add_done_callback(
        lambda x: processed_frames.extend(x.result())
      )
      frames = []
      frame_indices = []

    executor.submit(save_frame, frame_index)

    frame_index += 1
    
  # Process remaining frames
  if frames:
    executor.submit(process_batch, frames, frame_indices).add_done_callback(
      lambda x: processed_frames.extend(x.result())
    )
  
cap.release()
output_video.release()
