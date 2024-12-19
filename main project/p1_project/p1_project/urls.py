"""
URL configuration for p1_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from p1app.views import uhome,ulogin,usignup,ulogout,ubase

urlpatterns = [
    path('admin/', admin.site.urls),
    path("",uhome,name='uhome'),
    path("base/",ubase,name='ubase'),
    path("ulogin/",ulogin,name='ulogin'),
    path("usignup/",usignup,name='usignup'),
    path("ulogout/",ulogout,name='ulogout'),
]
