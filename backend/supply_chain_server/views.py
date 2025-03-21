from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import datetime  # 添加这行

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
            
            # 读取下一个 period 的假数据
            next_period = period + 1
            fake_data_path = f'/home/vislab/Yanjia/MAS_SupplyChain/backend/fake_data/env_period_{next_period}.json'
            
            with open(fake_data_path, 'r') as f:
                fake_data = json.load(f)
                return JsonResponse(fake_data, safe=False)
                
        except (json.JSONDecodeError, FileNotFoundError):
            return JsonResponse({"error": "Invalid data"}, status=400)
    else:
        return JsonResponse({"error": "Method not allowed"}, status=405)
