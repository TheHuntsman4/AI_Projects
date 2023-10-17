import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

model=YOLO('yolov8s.pt')
print(model.names) #this is the function to find out the labels
video=cv2.VideoCapture('cctvFootage.mp4')
count = 0

while True:
    ret,frame = video.read()
    count+=1
    if count%2!=0: #skipping even frames as it takes too long to iterate through everytihing
        continue
    frame=cv2.resize(frame,(1020,600))
    results=model.predict(frame)
    a=results[0].boxes.data #contains 5 columns with the last one being thr prediciton
    # print('this is', a)
    px=pd.DataFrame(a).astype("float")
    # print('this is the data frame:',px)
    for index,row in px.iterrows():
        # print(row)
        print(int(row[5])) #this is the prediction/ classification if is is 0 then it is a person
        print('\n')
        
    
    cv2.imshow('Output Prediction',frame)
    if cv2.waitKey(1)&0xFF==27: #condition when the video is finished
        break
    
video.release()
cv2.destroyAllWindows() 