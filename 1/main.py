import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import concurrent.futures

# Load the YOLOv8 model
model=YOLO('yolov8s.pt')

# Open the video file
video=cv2.VideoCapture('cctvFootage.mp4')

# Function to process a frame
def process_frame(frame):
    frame=cv2.resize(frame,(1020,600))
    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    person_count=0
    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        label=int(row[5])
        if label==0:
           person_count+=1
    cv2.putText(frame,'Number of people '+str(person_count),(500,300),cv2.FONT_HERSHEY_TRIPLEX,(0.5),(0,255,0),2)
    return frame

# Read all frames from the video and store them in a list
frames = []
count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    count += 1
    if count % 24 == 0:  # skipping even frames as it takes too long to iterate through everything
        frames.append(frame)

# Process the frames in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    processed_frames = list(executor.map(process_frame, frames))

# Display the processed frames
for frame in processed_frames:
    cv2.imshow('Output Prediction', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # condition when the video is finished
        break

video.release()
cv2.destroyAllWindows()
