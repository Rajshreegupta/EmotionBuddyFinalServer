from django.shortcuts import render
import socket
import cv2
from .models import FacialExpressionModel
import numpy as np

# Create your views here.
class Chatting:

    def __init__(self):
        self.reactionmessage=[]
        self.name={'192.168.0.13':"Rajshree"}
        self.messagelist=[]

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
        if addr[0] in self.name:
            self.clientname=self.name[addr[0]]
        self.messagelist.append(["Connected!!!", ""])
        self.reactionmessage.append("Talk to Know!!!")
        list=['Hi', 'Hello', 'Good Morning', 'Good Night']
        return render(request, 'home.html', {'clientname':self.clientname, 'messagelist': self.messagelist, 'reaction_of_client': self.reactionmessage, 'list': list})

    def sendmsg(self, request):
        message = request.POST["msg_from_server"]
        self.messagelist[-1][1]=message
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
        self.messagelist.append([incoming_message1, ""])

        incoming_message2 = self.conn.recv(1024)
        incoming_message2 = incoming_message2.decode()
        self.reactionmessage.append(incoming_message2)

        if incoming_message2 == "angry":
            list = ['Dont lose patience, everything shall be fine', 'I understand your disagreement, but have patience',
                    'Anger is not good for health']

        elif incoming_message2 == "disgust":
            list = ['I am sorry if I offended you', 'I did not mean that', 'I am sorry']

        elif incoming_message2 == "afraid":
            list = ['Have faith in God', 'Do you want me next to you', 'Are you okay? You can share with me.']
        elif incoming_message2 == "happy":
            list = ['May I also know the joke', 'Very happy for you']
        elif incoming_message2 == "neutral":
            list = ['Good Morning', 'Good night', 'Hi', 'Hello']
        elif incoming_message2 == "sad":
            list = ['Cheer up dear, everything shall be fine', 'Your mood really effects me',
                    'Cannot see you like that', 'Tell what I can do for you']
        else:
            list = ['Its too good but yes true', 'You read it correct :)']

        return render(request, 'home.html', {'clientname': self.clientname, 'messagelist': self.messagelist,
                                             'reaction_of_client': self.reactionmessage, 'list':list})
