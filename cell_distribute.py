#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:45:03 2019

@author: wuzhiqiang
#通用的电压一致性聚类判断和pca降维画图
"""
print(__doc__)

import app_path
import os, sys
import rw_data as rwd
import process_data as ppd
import model as md
import func as fc
import time, datetime
import numpy as np
np.set_printoptions(suppress=True) #不用科学计数法显示

def init_data_para():
    para_dict = {}
    para_dict['data_dir'] = {'debug': os.path.normpath('./zc/data'),
                             'run': os.path.normpath('/raid/exchange_data/zhongche/cell_distribute')}
    para_dict['pkl_dir'] = {'debug': os.path.normpath('./zc/pkl'),
                             'run': os.path.normpath('/raid/exchange_data/zhongche/cell_distribute/pkl')}
    para_dict['config'] = {'debug': {'s': '192.168.1.105', 'u': 'data', 'p': 'For2019&tomorrow', 'db': 'bsa', 'port': 3306},
                             'run': {'s': 'localhost', 'u': 'data', 'p': 'For2019&tomorrow', 'db': 'bsa', 'port': 3306}
                             }
    para_dict['od_table_name'] = 'cell_calculation'#'cell_meanAndstd_yz'
    para_dict['info_table_name'] = 'vehicle_info'
    para_dict['window_len'] = {'debug': 10,
                                 'run': 12}
    para_dict['all_len'] = {'debug': 15,
                                 'run': 500}
    para_dict['t1_period'] = {'debug': 1, 'run': 3600}
    para_dict['t2_period'] = {'debug':5, 'run': 3600 * 24 * 60} #2个月
    para_dict['cycle'] = {'debug': 5, 'run': 94608000}#3years
    
    para_dict['outliers_fraction'] = {'mean': 0.072,
                                         'std': 0.072}
    
    para_dict['vin'] = 'vehicle_id'
    para_dict['pack_id'] = 'bp_id'
    para_dict['bat_fac'] = 'vehicle_model'
    para_dict['time'] = 'end_time'
    para_dict['mean'] = 'average'
    para_dict['std'] = 'standard_deviation'
    para_dict['columns'] = ['vehicle_id', 'bp_id', 'end_time', 'pca_average', 'pca_standard_deviation', 'clf_pca_average', 'clf_pca_standard_deviation']

    para_dict['run_mode'] = 'debug'
    return para_dict

def reg_outliers_fraction(outliers_fraction, key):
    if key == 'optimum':
        outliers_fraction -= 0.003
    return outliers_fraction

def build_model(para_dict, mode):
    """
    将所有车辆数据按电池类型划分后，进行建模
    考虑定期如1个月做一次
    一辆车一天运行10小时，半小时一条数据，则一个月大概20*30条数据,读取的数据长度可以设置在500条以内
    假设2个月电池衰减后，异常电池的比例增加0.15%，则outliers_fraction每次加0.0015，3年后增加总量为0.03左右
    人为添加的异常值暂不考虑改变
    """
    while True:
        base_para = yield
        delta_outliers_fraction = 0.0015 * base_para
        para_dict['outliers_fraction']['mean'] += delta_outliers_fraction
        para_dict['outliers_fraction']['std'] += delta_outliers_fraction
        limit_len = para_dict['all_len'][mode]
        interfere_data = [0, 10, 2, 15]
        #读取所需的数据，并进行处理后存储到指定位置
        print('starting building the model...')
        classifiers = {}
        vin_info = rwd.get_vin_list(para_dict, mode)
        if vin_info is not None:
            vin_gp = vin_info.groupby(para_dict['bat_fac'])
            for key in vin_gp.groups.keys():
                classifiers[key] = []
                first = True
                vin_list = vin_gp.get_group(key)['vin'].tolist()
                for vin in vin_list:
                    sort_data = rwd.read_sort_data(para_dict, mode, vin, limit=limit_len)
                    if sort_data is None:
                        print('there is no data in table %s.'%vin)
                        continue
                    sort_data, mean_pca_arr = ppd.get_pca(sort_data, para_dict['mean'], *interfere_data)
                    sort_data, std_pca_arr = ppd.get_pca(sort_data, para_dict['std'], *interfere_data)
                    if first:
                        all_mean_pca_arr = mean_pca_arr #创建同一电池类型的所有pca后的数据空间
                        all_std_pca_arr = std_pca_arr
                        first = False
                    else:
                        all_mean_pca_arr = np.r_[all_mean_pca_arr, mean_pca_arr]#添加同一电池类型的所有pca后的数据
                        all_std_pca_arr = np.r_[all_std_pca_arr, std_pca_arr]
                outliers_fraction = reg_outliers_fraction(para_dict['outliers_fraction']['mean'], key)
                mean_clf_model= md.build_model(para_dict, mode, key, all_mean_pca_arr, 'mean', outliers_fraction)
                outliers_fraction = reg_outliers_fraction(para_dict['outliers_fraction']['std'], key)
                std_clf_model= md.build_model(para_dict, mode, key, all_std_pca_arr, 'std', outliers_fraction)
                classifiers[key].append(mean_clf_model)
                classifiers[key].append(std_clf_model)
            
            print('the models have been created by %d times.'%int(base_para))
        else:
             print('there is no vehicle!')

def predict(para_dict, mode):
    mean_clf_model = None
    std_clf_model = None
    classifiers = {}
    interfere_data = [0, 1, 1, 0]
    while True:
        print('waiting for the notice which from the timer...')
        c = yield
        print('starting predit the result...')
        vin_info = rwd.get_vin_list(para_dict, mode)
        if vin_info is not None:
            vin_gp = vin_info.groupby(para_dict['bat_fac'])
            for key in vin_gp.groups.keys():
                mean_clf_model, std_clf_model = rwd.get_model(classifiers, key, para_dict, mode)
                vin_list = vin_gp.get_group(key)['vin'].tolist()
                for vin in vin_list:
                    sort_data = rwd.read_sort_data(para_dict, mode, vin, limit=para_dict['window_len'][mode])
                    if sort_data is None:
                        print('there is no data in table %s.'%vin)
                        continue
                    sort_data, mean_pca_arr = ppd.get_pca(sort_data, para_dict['mean'], *interfere_data)
                    sort_data, std_pca_arr = ppd.get_pca(sort_data, para_dict['std'], *interfere_data)
                    sort_data = md.outlier_predict(mean_clf_model, sort_data, 'pca_'+para_dict['mean'], mode)
                    sort_data = md.outlier_predict(std_clf_model, sort_data, 'pca_'+para_dict['std'], mode)
                    res = sort_data[para_dict['columns']]
                    rwd.save_distribute_data(res, vin, para_dict, mode)#存储处理后的数据
        else:
            print('there is no vehicle!')
         
def timer(para_dict, mode, clf, model):
    """
    计时器，产生两个计时器，分别触发build_model和predict
    按照3年周期，build每2个月运行一次，会运行18次，将每次运行的次数传入build_model，用作调整加入异常值的基数
    """
    max_ticks = para_dict['t2_period'][mode] // para_dict['t1_period'][mode] #两个计时器间隔
    ticks = 0
    base_para = 0
    next(model)
    model.send(base_para)
    next(clf)
    while para_dict['cycle'][mode] >= 0:
        para_dict['cycle'][mode] -= 1
        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('predict: %s'%now_time)
        clf.send(ticks)
        ticks += 1
        if ticks == max_ticks:
            print('build model: %s'%now_time)
            model.send(base_para)
            base_para += 1
            ticks = 0
        time.sleep(para_dict['t1_period'][mode])
    
def main(argv):
    
    para_dict =  init_data_para()
    para_dict = fc.deal_argv(argv, para_dict)
    mode = para_dict['run_mode']
    model = build_model(para_dict, mode)
    clf = predict(para_dict, mode)
    timer(para_dict, mode, clf, model)
    
    
if __name__ == '__main__':
    main(sys.argv)