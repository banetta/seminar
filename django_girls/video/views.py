# video/views.py
from django.shortcuts import render, redirect
from django.core.urlresolvers import reverse
from .models import Video​
​
​
def video_list(request):
    video_list = Video.objects.all()
    return render(request, 'video/video_list.html', {'video_list': video_list})