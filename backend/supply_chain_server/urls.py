"""
URL configuration for supply_chain_server project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
from django.urls import path, re_path
from django.views.static import serve
from django.views.generic import TemplateView
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from . import views
import json
import os

def serve_json_file(request, path):
    try:
        # 尝试从 dist/test_data 目录读取
        json_path = settings.BASE_DIR / 'dist' / 'test_data' / path
        if not os.path.exists(json_path):
            return JsonResponse({'error': f'File not found: {path}'}, status=404)
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return JsonResponse(data, safe=False)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON file'}, status=500)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def serve_js_file(request, path):
    try:
        js_path = settings.BASE_DIR / 'dist/lang' / path
        if not os.path.exists(js_path):
            return HttpResponse(status=404)
            
        with open(js_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return HttpResponse(content, content_type='application/javascript')
    except Exception as e:
        return HttpResponse(str(e), status=500)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('hello/', views.hello_world, name='hello_world'),
    path('print_post/', views.print_post, name='print_post'),
    path('next_step/', views.next_step, name='next_step'),
    path('chat/', views.chat, name='chat'),
    
    # 添加数据文件路由（移到静态文件路由前面）
    re_path(r'^test_data/(?P<path>.*)$', serve_json_file),
    
    # 静态文件服务
    re_path(r'^assets/(?P<path>.*)$', serve, {
        'document_root': settings.BASE_DIR / 'dist/assets'
    }),
    
    # 添加图片文件路由
    re_path(r'^imgs/(?P<path>.*)$', serve, {
        'document_root': settings.BASE_DIR / 'dist/imgs'
    }),
    re_path(r'^gif/(?P<path>.*)$', serve, {
        'document_root': settings.BASE_DIR / 'dist/gif'
    }),
    
    # 添加语言文件路由
    # 修改语言文件路由，使用 serve_js_file 处理函数
    re_path(r'^lang/(?P<path>.*)$', serve_js_file),  # 注意这里改成 lang 而不是 langdist
    
    # 删除或注释掉这个旧的路由
    # re_path(r'^langdist/(?P<path>.*)$', serve, {
    #     'document_root': settings.BASE_DIR / 'dist/lang'
    # }),
    
    # 添加 i18n 语言文件路由
    re_path(r'^locales/(?P<path>.*)$', serve, {
        'document_root': settings.BASE_DIR / 'dist/locales'
    }),
    
    # 确保这些路由在通配符路由之前
    path('', TemplateView.as_view(template_name='index.html')),
    re_path(r'^.*$', TemplateView.as_view(template_name='index.html')),
]
