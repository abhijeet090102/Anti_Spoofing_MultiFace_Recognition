from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
app_name = 'accounts'

urlpatterns = [
    path('register/', views.register_user, name='register'),
    path('admin_dashboard/', views.admin_dashboard),
    path('', views.login, name='login'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('stop_video/', views.stop_video, name='stop_video'),
    path('register_data/', views.User_register, name='register_data'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)