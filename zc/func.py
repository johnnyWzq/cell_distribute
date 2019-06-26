#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:54:14 2019

@author: wuzhiqiang
"""

import lib_path
import re

def deal_argv(argv, para_dict):
    """
    第一位代表运行模式，debug/run
    """
    para_len = len(argv)
    if para_len  >= 2:
        import g_function as gf
        para_dict['run_mode'] = gf.deal_1_argv(argv)    
        if para_len > 2:
            print("dealing others parameters...")
            para_dict = deal_other_argv(argv, para_dict)  
    return para_dict

def deal_other_argv(argv, para_dict):
    """
    1:bat_mode; 2:bat_type; 3:bat-structure; 4:bat-year; 5:score_key
    """
    i = 0
    for ar in argv:
        if i == 2: #bat_mode
            regx = '-[0-9]{1,2}'
            if re.match(regx, ar):
                para_dict['window_len'] = ar[1:]
                print('The 2nd input parameter is accepted.')
            else:
                print("The 2nd input parameter '%s' is not accepted."%ar)
        i += 1
    return para_dict