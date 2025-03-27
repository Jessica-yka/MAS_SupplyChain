#!/bin/bash

# 使用完整路径激活虚拟环境
. /home/vislab/Yanjia/MAS_SupplyChain/uist2025/bin/activate

# 使用完整路径执行 Django 命令
python /home/vislab/Yanjia/MAS_SupplyChain/backend/manage.py collectstatic --noinput
python /home/vislab/Yanjia/MAS_SupplyChain/backend/manage.py runserver 0.0.0.0:8000
