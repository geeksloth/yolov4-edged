import os
import argparse
from time import time
import logsging as logs
import cv2
import numpy as np
import json
import queue
import threading
from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import base64


def check_allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

def pprint(mydict):
	print(json.dumps(mydict, sort_keys=True, indent=4))

class VideoCapture2:
	def __init__(self, name):
		self.cap = cv2.VideoCapture(name)
		self.q = queue.Queue()
		t = threading.Thread(target=self._reader)
		t.daemon = True
		t.start()
	def _reader(self):
		while True:
			ret, frame = self.cap.read()
			if not ret:
				break
			if not self.q.empty():
				try:
					self.q.get_nowait()
				except queue.Empty:
					pass
			self.q.put(frame)
	def read(self):
		return self.q.get()

parser = argparse.ArgumentParser()
parser.add_argument(
	"-c", "--confidence", type=float, default=0.05,	help="confidence threshold"
)
parser.add_argument(
	"-nms", "--non_maximum", type=float, default=0.5, help="non-maximum supression threshold"
)
parser.add_argument(
	"-ll", "--logsging_level", type=str, default=30, help="logsging level"
)
parser.add_argument(
	"-gpu", "--gpu", type=bool, default=True, help="enable GPU"
)
args = vars(parser.parse_args())
logs.basicConfig(level=args["logsging_level"])
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
classes = list()
with open("resources/labels.txt", "r") as f:
	classes = [cname.strip() for cname in f.readlines()]
NN = cv2.dnn.readNetFromDarknet(
	"resources/yolov4-tiny.cfg",
	"resources/yolov4-tiny.weights"
)
if args["gpu"]:
	NN.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	NN.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	cv2.cuda.setDevice(0)
NN_layer = NN.getLayerNames()
NN_layer = [NN_layer[i - 1] for i in NN.getUnconnectedOutLayers()]
allowed_exts = {'jpg', 'jpeg','png','JPG','JPEG','PNG'}
app = Flask(__name__)

@app.route("/",methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		if 'file' not in request.files:
			print('No file attached in request')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			print('No file selected')
			return redirect(request.url)
		if file and check_allowed_file(file.filename):
			stats = list()
			founds = list()
			boxes = []
			classIds = []
			confidences = []
			filename = secure_filename(file.filename)
			img = Image.open(file.stream)
			if img.mode != 'RGB':
				img = img.convert('RGB')
			open_cv_image = np.array(img) 
			# Convert RGB to BGR 
			frame = open_cv_image[:, :, ::-1].copy() 
			W, H = img.size
			NN.setInput(
				cv2.dnn.blobFromImage(
					frame, 
					1/255.0, (416, 416),
					swapRB=True, 
					crop=False
				)
			)
			for output in NN.forward(NN_layer):
				for detection in output:
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]
					if confidence > args["confidence"]:
						box = detection[0:4] * np.array([W, H, W, H])
						(center_x, center_y, width, height) = box.astype("int")
						x = int(center_x - (width/2))
						y = int(center_y - (height/2))
						boxes.append([x, y, int(width), int(height)])
						classIds.append(classID)
						confidences.append(float(confidence))
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["non_maximum"])
			if len(idxs) > 0:
				for i in idxs.flatten():
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])
					founds.append("{},[{},{},{},{}],{:.2f}".format(
						classes[classIds[i]],
						x,y,w,h,confidences[i]
					))
					text = "{}".format(classes[classIds[i]])
					cv2.putText(frame, text, (x+(int)(w/2), y+(int)(h/2)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			retval, buffer = cv2.imencode('.jpg', frame)
			encoded_string = base64.b64encode(buffer).decode()  
		return render_template('index.html', img_data=encoded_string, json_data=founds), 200
	else:
		return render_template('index.html', img_data="", json_data=""), 200
if __name__ == "__main__":
	app.debug=True
	app.run(host='0.0.0.0', port=5002)