# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import os
python_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(python_path)
os.environ["PYTHONPATH"] = python_path