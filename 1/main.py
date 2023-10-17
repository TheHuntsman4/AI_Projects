import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

model=YOLO('yolov8s.pt')
# print(model.names) #this is the function to find out the labels
video=cv2.VideoCapture('cctvFootage.mp4')
count = 0

while True:
    ret,frame = video.read()
    count+=1
    if count%2!=0: #skipping even frames as it takes too long to iterate through everytihing
        continue
    frame=cv2.resize(frame,(1020,600))
    results=model.predict(frame)
    a=results[0].boxes.data #contains 5 columns with the last one being the prediciton
    # print('this is a', a)
    px=pd.DataFrame(a).astype("float")
    # print('this is the data frame:',px)
    person_count=0
    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        label=int(row[5])
        if label==0:
           person_count+=1
           cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
           cv2.putText(frame,str('person'),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,(0.5),(255,255,255),1)
        
    cv2.putText(frame,'Number of people '+str(person_count),(510,400),cv2.FONT_HERSHEY_TRIPLEX,(0.5),(0,255,0),2)
    cv2.imshow('Output Prediction',frame)
    if cv2.waitKey(1)&0xFF==27: #condition when the video is finished
        break
    
video.release()
cv2.destroyAllWindows() 