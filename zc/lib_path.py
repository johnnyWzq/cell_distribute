#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:26:42 2019

@author: wuzhiqiang
"""
import sys,os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))#存放c.py所在的绝对路径
lib_dir = "libs"
level = 2
while level > 0:
    BASE_DIR = os.path.dirname(BASE_DIR)
    level -= 1
lib_dir = os.path.join(BASE_DIR, lib_dir)
if lib_dir not in sys.path:
    sys.path.append(lib_dir)