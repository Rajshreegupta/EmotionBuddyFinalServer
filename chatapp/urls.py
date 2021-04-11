from django.contrib import admin
from django.urls import path, include
from .views import Chatting

chat=Chatting()

urlpatterns = [
    path('', chat.home, name='home'),
    path('sendmsg', chat.sendmsg, name='sendmsg')
]