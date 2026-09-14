from django.urls import path
from . import views

urlpatterns = [
    path("",               views.ask_view,    name="ask"),
    path("stream-answer/", views.stream_view, name="stream_answer"),
]