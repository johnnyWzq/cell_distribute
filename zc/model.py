#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:13:13 2019

@author: wuzhiqiang
"""

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
#from scipy import stats
from sklearn.externals import joblib
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

#outliers_fraction = 0.075

def build_model(para_dict, mode, bat_type, pca_arr, data_type, outliers_fraction):
    #outliers_fraction = para_dict['outliers_fraction'][data_type]
    X_train = pca_arr
    #clf = svm.OneClassSVM(nu=outliers_fraction, kernel="rbf")
    clf = EllipticEnvelope(contamination=outliers_fraction)
    #clf = IsolationForest(contamination=outliers_fraction)
    #for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(X_train)
    model_time = datetime.datetime.now().strftime('%Y-%m-%d')
    pkl_dir = para_dict['pkl_dir'][mode]
    model_name = bat_type + '_' + str(data_type) + '_' + model_time
    save_model(clf, pkl_dir, model_name)
    if mode == 'debug':
        y_pred_train = clf.predict(X_train)
        n_error_train = y_pred_train[y_pred_train == -1].size
        #scores_pred = clf.decision_function(X_train)
        #threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
        #draw_res(clf, threshold, X_train, bat_type, False)
        print('outliers_fraction=%.4f'%outliers_fraction)
        print('n_error_train: %d'%n_error_train)
        print('')
        draw_res(clf, 0, X_train, bat_type+'_'+data_type, False)
    return clf

def outlier_predict(model, df, col, mode):
    y_pred_test_list = []
    X_test_list = []
    for i in range(len(df)):
        X_test = df[col].iloc[i]
        y_pred_test = model.predict(X_test)
        n_error_test = y_pred_test[y_pred_test == -1].size
        print(n_error_test, end=', ')
        y_pred_test_list.append(y_pred_test.tolist())
        X_test_list.append(X_test)
    print('')
    df['clf_'+col] = y_pred_test_list
    if mode == 'debug':
        threshold = 0
        draw_res(model, threshold, X_test_list, df['vehicle_id'].iloc[0])
    return df

def save_model(model, pkl_dir, pkl_name):
    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)
    joblib.dump(model, os.path.join(pkl_dir, pkl_name+'.pkl'))
    
def load_model(model_name, pkl_dir):
    model = joblib.load(os.path.join(pkl_dir, model_name))
    return model

def draw_res(clf, threshold, x_list, title_name, is_list=True):
    xx, yy = np.meshgrid(np.linspace(-10, 30, 500), np.linspace(-10, 30, 500))
    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.PuBu)  #绘制异常点区域，值从最小的到阈值的那部分
    plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='darkred') #绘制异常点区域和正常点区域的边界
    #plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='palevioletred') #绘制正常点区域，值从阈值到最大的那部分
    if is_list:
        draw_scatter_list(x_list, title_name)
    else:
        draw_scatter(x_list, title_name)
    plt.show()
    
def draw_scatter_list(data_list, title_name):
    if data_list is None:
        return
    
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    
    i = 0
    for data in data_list:
        marker = 'o'
        plt.scatter(data[:, 0], data[:, 1],marker=marker, alpha=0.65)
        plt.title(str(title_name))
        i += 1
    #plt.show()

def draw_scatter(data, title_name):
    if data is None:
        return
    
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

    marker = 'o'
    plt.scatter(data[:, 0], data[:, 1],marker=marker, alpha=0.35)
    plt.title(str(title_name))