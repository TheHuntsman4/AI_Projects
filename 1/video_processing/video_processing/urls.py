from django.urls import path
from video_app.views import index, upload_video

urlpatterns = [
    path('', index, name='index'),
    path('main', upload_video, name='upload_video'),
]
