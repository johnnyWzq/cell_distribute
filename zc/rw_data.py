#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:01:53 2019

@author: wuzhiqiang
"""
import time
import re

import lib_path
import io_operation as ioo
import g_function as gf
import model as md


def get_vin_list(para_dict, mode):
    config = para_dict['config'][mode]
    vin_info = ioo.read_sql_data(config, para_dict['info_table_name'])
    print(vin_info)
    return vin_info

def read_sort_data(para_dict, mode, vin, limit=None):
    config = para_dict['config'][mode]
    sort_data = ioo.match_sql_data(config, para_dict['od_table_name'], para_dict['vin'], vin, limit)
    return sort_data

def save_distribute_data(data, filename, para_dict, mode, chunksize=None):
    for col in para_dict['columns']:
        if 'pca' in col:
            data[col] = data[col].apply(str)
            data[col] = data[col].apply(lambda x: x.replace('\n', ''))
            data[col] = data[col].apply(lambda x: x.replace(',', ';'))
    ioo.save_data_csv(data, filename, para_dict['data_dir'][mode])
    
def get_model(classifiers, key, para_dict, mode):
    if key in classifiers.keys():
        if classifiers[key] is not None:
            mean_clf_model, std_clf_model = classifiers[key][0], classifiers[key][1]
        else:
            mean_clf_model = read_pkl(para_dict['pkl_dir'][mode], key, 'mean')
            std_clf_model = read_pkl(para_dict['pkl_dir'][mode], key, 'std')
    else:
        mean_clf_model = read_pkl(para_dict['pkl_dir'][mode], key, 'mean')
        std_clf_model = read_pkl(para_dict['pkl_dir'][mode], key, 'std')
    return mean_clf_model, std_clf_model

def read_pkl(pkl_dir, key1, key2):
    regx = r'(%s)\_(%s)\_([0-9\-]{8,12})(\.pkl)'%(key1, key2)
    file_names = ioo.get_all_files_name(pkl_dir, regx)
    time_list = []
    for file_name in file_names:
        m = re.match(regx, file_name)
        time_list.append(time.mktime(time.strptime(m.group(3), '%Y-%m-%d'))) #转换为struct_time
    index = gf.get_recent_time(time_list)
    file_name = file_names[index]
    model = md.load_model(file_name, pkl_dir)
    return model
#read_pkl('./pkl', 'catl', 'mean')