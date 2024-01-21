import cv2
import torch
import numpy as np
from tracker import *


model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

cap=cv2.VideoCapture('highway.mp4')

count=0
tracker = Tracker()




def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

while True:
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    
    frame=cv2.resize(frame,(1020,600))
    results=model(frame)
    veihcle_list=[]
    for index, rows in (results.pandas().xyxy[0].iterrows()):
        x=int(rows[0])
        y=int(rows[1])
        x1=int(rows[2])
        y1=int(rows[3])
        class_name=str(rows['name'])
        veihcle_list.append([x,y,x1,y1])
    idx_bbox=tracker.update(veihcle_list)
    for bbox in idx_bbox:
        x2,y2,x3,y3,id=bbox
        cv2.rectangle(frame,(x2,y2),(x3,y3),(0,0,255),2)
        cv2.putText(frame,str(id),(x2,y2),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
        
    
    cv2.imshow("FRAME",frame)
    if cv2.waitKey(0)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()