import threading
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import io
from django.http import StreamingHttpResponse, HttpResponseRedirect
from django.shortcuts import render

class ProcessVideo(threading.Thread):
    def __init__(self, video_file, callback):
        threading.Thread.__init__(self)
        self.video_file = video_file
        self.callback = callback

    def run(self):
        video_filename = 'temp_video.mp4'
        with open(video_filename, 'wb') as f:
            for chunk in self.video_file.chunks():
                f.write(chunk)

        video = cv2.VideoCapture(video_filename)
        model = YOLO('yolov8s.pt')
        count = 0

        while True:
            ret, frame = video.read()
            count += 1
            if count % 24 != 0:
                continue
            if not ret:
                self.callback()  # Trigger the callback function when the video ends
                break
            frame = cv2.resize(frame, (1020, 600))
            results = model.predict(frame)
            a=results[0].boxes.data
            px = pd.DataFrame(a).astype("float")
            person_count = 0
            for index, row in px.iterrows():
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                label = int(row[5])
                if label == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, str('person'), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    person_count += 1
            cv2.putText(frame, 'Number of people: ' + str(person_count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)
            _, encodedImage = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + encodedImage.tobytes() + b'\r\n')

def index(request):
    return render(request, 'index.html')

def end(request):
    return render(request, 'end.html')

def upload_video(request):
    if request.method == 'POST':
        video_file = request.FILES['file']
        new_thread = ProcessVideo(video_file, callback=lambda: HttpResponseRedirect('/end'))
        new_thread.start()
        return StreamingHttpResponse(new_thread.run(), content_type='multipart/x-mixed-replace; boundary=frame')
    return render(request, 'upload.html')
