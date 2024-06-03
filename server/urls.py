from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

from django.conf import settings
from django.conf.urls.static import static

router = DefaultRouter()
router.register(r'main_objects', views.MainObjectViewSet)
router.register(r'stream', views.MainObjectViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('api/get_response', views.MainObjectViewSet.as_view({
        'post': 'create_with_question_answer',
        'put': 'update_with_question_answer'
    })),
    path('api/species_detail', views.MainObjectViewSet.as_view({
        'get': 'getSpeciesDetail'
    })),
    path('api/upload_image', views.MainObjectViewSet.as_view({
        'post':'upload_image'
        })),
    path('api/upload_video', views.MainObjectViewSet.as_view({
        'post':'upload_video'
        })),
    path('api/generate_pattern', views.MainObjectViewSet.as_view({
        'post':'generate_pattern'
        })),
    path('api/segment_image', views.MainObjectViewSet.as_view({
        'post':'segment_image'
        })),
    path('api/event-stream', views.PostStreamView.as_view(), name='stream'),
]
