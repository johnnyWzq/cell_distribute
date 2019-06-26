#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:25:32 2019

@author: wuzhiqiang
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import re

def get_2_data(text, split):
    #[[]]
    x = text[2:-2].split(split)
    return x
    
def get_data(text,regx):
    #将字符串划分得到数据
    #[]
    pattern = re.compile(regx)
    num_regx = pattern.findall(text)
    x = list(map(float, [x for x in num_regx]))
    return x

def trans_data(df):
    if df.empty:
        return None
    new_data_list = []
    for num, v_mean in df.items():
        rows = get_2_data(v_mean, '],[')
        sv_list = []
        for row in rows:
            regx =  r'[0-9]{0,3}\.[0-9]{0,2}'
            row_data = get_data(row, regx)
            sv_list.append(row_data)
        column = ['no' + str(i) for i in range(len(row_data))]
        row_data = pd.DataFrame(sv_list, columns=column)  #得到一组排名数据
        new_data_list.append(row_data)
    #data = pd.concat(tuple(new_data_list))
    return new_data_list

def add_interfere(df, *arg):
    #rows改变的行数，index改变的列数，列默认从0开始
    start_row = arg[0]
    row_num = arg[1]
    index = arg[2]
    value = arg[3]
    start_row = min(df.shape[0]-row_num, start_row)
    index = min(df.shape[1], index)
    df.iloc[start_row:start_row+row_num, 0: index] += value 
    return df

def add_interfere_pca(df_list, *arg):
    new_df_list = []
    for df in df_list:
        new_df_list.append(data_pca(add_interfere(df, *arg)))
    return new_df_list

def data_pca(row_data, whiten=False):
    pca = PCA(n_components=2, whiten=whiten)
    pca.fit(row_data)
    #print(pca.explained_variance_)
    #print(pca.explained_variance_ratio_)
    new_data = pca.transform(row_data)
    new_data = np.around(new_data, 3)
    return new_data

def get_pca(sort_data, col, *arg):
    data_list = trans_data(sort_data[col])
    data_pca_list = add_interfere_pca(data_list, *arg)
    sort_data['pca_'+col] = data_pca_list #将降维后的数据片写回sort_data
    #将所有的数据组合成一个array
    data_pca_arr = data_pca_list[0]
    for i in range(1, len(data_pca_list)):
        data_pca_arr = np.r_[data_pca_arr, data_pca_list[i]]
    
    return sort_data, data_pca_arr