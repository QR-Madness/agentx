from django.urls import path

from . import views

urlpatterns = [
    path("index", views.index, name="index"),
    path("language-detect", views.language_detect, name="language-detect"),
]