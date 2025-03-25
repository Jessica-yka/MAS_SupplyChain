from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import datetime
import time
import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(current_dir)

# 现在可以导入主模块了
from main_single_period import period_simulation_framework, ENV_CONFIG_NAME

def hello_world(request):
    return HttpResponse("Hello World")

@csrf_exempt
def print_post(request):
    if request.method == 'POST':
        try:
            # 获取POST的数据
            data = json.loads(request.body)
            # 打印到控制台
            print("Received POST data:", data)
            # 返回响应
            return JsonResponse({
                "status": "success",
                "message": "Data received",
                "data": data
            })
        except json.JSONDecodeError:
            return JsonResponse({
                "status": "error",
                "message": "Invalid JSON data"
            }, status=400)
    else:
        return JsonResponse({
            "status": "error",
            "message": "Only POST method is allowed"
        }, status=405)

@csrf_exempt
def next_step(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            period = data.get('period')
            agents = data.get('agents')
            events = data.get('events')
            
            # 打印请求信息
            print(f"收到请求 - 时间: {datetime.datetime.now()}")
            print(f"请求数据: {agents}")
            print(f"当前周期: {period}")
            print(f"当前事件: {events}")

            # 开发模式的假数据
            fake_data_path = f'/home/vislab/Yanjia/MAS_SupplyChain/backend/fake_data/env_period_1.json'
            result = json.load(fake_data_path)

            # 运行模式
            # if period == -1:
            #     # 初始化模拟
            #     result = period_simulation_framework(env_json=None, cur_period=0)
            #     # 读取初始化后的环境数据
            # else:
            #     # 使用当前环境数据继续模拟
            #     result = period_simulation_framework(env_json=agents, cur_period=period+1)
            
            return JsonResponse(result, safe=False)
                
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"请求处理出错: {str(e)}")
            return JsonResponse({"error": "Invalid data"}, status=400)
    else:
        return JsonResponse({"error": "Method not allowed"}, status=405)
