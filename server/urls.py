from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

from django.conf import settings
from django.conf.urls.static import static
from .views import PostStreamView

router = DefaultRouter()
router.register(r'main_objects', views.MainObjectViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('get_response', views.MainObjectViewSet.as_view({
        'post': 'create_with_question_answer',
        'put': 'update_with_question_answer'
    })),
    path('species_detail', views.MainObjectViewSet.as_view({
        'get': 'getSpeciesDetail'
    })),
    path('event-stream', PostStreamView.as_view(), name='stream'),
]
