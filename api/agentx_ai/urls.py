from django.urls import path

from . import views

urlpatterns = [
    path("index", views.index, name="index"),
    path("tools/language-detect-20", views.language_detect, name="language-detect"),
    path("tools/translate", views.translate, name="translate"),
]