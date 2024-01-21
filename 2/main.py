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
area1=[(367,437),(324,471),(526,482),(521,438)]
in_area_1=set()
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
        cv2.circle(frame,(x3,y3),3,(0,255,0),-1)
        result=cv2.pointPolygonTest(np.array(area1,np.int32),((x3,y3)),False)
        if result>0:
            in_area_1.add(id)
        
    cv2.polylines(frame, [np.array(area1,np.int32)],True,(0,255,255),3)
    cv2.imshow("FRAME",frame)
    print(in_area_1)
    if cv2.waitKey(1)&0xFF==27:
        break
    print(f'The number of vehicles leaving the frame: {len(in_area_1)}')
cap.release()
cv2.destroyAllWindows()