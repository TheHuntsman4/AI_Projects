import threading
from flask import Flask, request, Response, render_template
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import io

app = Flask(__name__)

class ProcessVideo(threading.Thread):
    def __init__(self, video_file):
        threading.Thread.__init__(self)
        self.video_file = video_file

    def run(self):
        video_filename = 'temp_video.mp4'
        self.video_file.save(video_filename)

        video = cv2.VideoCapture(video_filename)
        model = YOLO('yolov8s.pt')
        count = 0

        while True:
            ret, frame = video.read()
            count += 1
            if count % 2 != 0:
                continue
            frame = cv2.resize(frame, (1020, 600))
            results = model.predict(frame)
            a = results.xyxy[0].numpy()
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
                        (255, 255, 255), 2)
            _, encodedImage = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + encodedImage.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    video_file = request.files['file']
    new_thread = ProcessVideo(video_file)
    new_thread.start()
    return Response(new_thread.run(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
