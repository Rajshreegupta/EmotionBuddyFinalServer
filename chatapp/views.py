from django.shortcuts import render
import socket
import cv2
from .models import FacialExpressionModel
import numpy as np

# Create your views here.
class Chatting:
    def home(self, request):
        self.facec = cv2.CascadeClassifier('static/jsfile/haarcascade_frontalface_default.xml')
        self.model = FacialExpressionModel()
        self.cam = cv2.VideoCapture(0)

        cv2.namedWindow("test")

        s=socket.socket()
        host = socket.gethostname()
        port = 8080
        s.bind((host, port))
        s.listen(1)
        self.conn, addr = s.accept()
        return render(request, 'home.html', {'msg_from_client': 'Client connect!!!'})

    def sendmsg(self, request):
        message = request.POST["msg_from_server"]
        message = message.encode()
        self.conn.send(message)

        ret, frame = self.cam.read()

        gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pred = "Default"
        faces = self.facec.detectMultiScale(gray_fr, 1.3, 5)
        for (x,y,w,h) in faces:
            fc = gray_fr[y:y+h, x:x+h]

            roi = cv2.resize(fc, (48, 48))
            pred = self.model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        msg = pred

        msg=msg.encode()
        self.conn.send(msg)

        incoming_message1 = self.conn.recv(1024)
        incoming_message1 = incoming_message1.decode()

        incoming_message2 = self.conn.recv(1024)
        incoming_message2 = incoming_message2.decode()

        return render(request, 'home.html', {'msg_from_client': incoming_message1, 'reaction_of_client': incoming_message2})
