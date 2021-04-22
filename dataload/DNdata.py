import numpy as np
import pandas as pd
import xlrd
import torch
import pdb

def load_data(batch_size, train = True):
    loadcurve_data = []
    if train: 
        data = xlrd.open_workbook(r"G:\论文\GAN\AnoGAN-lhb\dataload\normal_data.xls") #读取文件
        shuffle = True
    else:
        data = xlrd.open_workbook(r"G:\论文\GAN\AnoGAN-lhb\dataload\test_data.xls") #读取文件
        shuffle = False
    table = data.sheet_by_index(0)   # 按索引获取工作表，0就是工作表1
    for i in range(table.nrows):     # table.nrows表示总行数
        line = table.row_values(i)   # 读取每行数据，保存在line里面，line是list
        loadcurve_data.append(line)        # 将line加入到resArray中，resArray是二维list
    loadcurve_data = np.array(loadcurve_data)    # 将resArray从二维list变成数组 [28800, 144]
    data_cnt = int(loadcurve_data.shape[0])
    loadcurve_data = loadcurve_data.astype(np.float32)

    mean_value = np.mean(loadcurve_data)
    max_value = np.abs(loadcurve_data).max()
    loadcurve_data = (loadcurve_data - mean_value) / max_value  # 归一化处理 [-1,1]之间
    loadcurve_data_img = np.expand_dims(loadcurve_data.reshape([-1, 12, 12]), axis = 1) # [28800, 1, 12, 12]
    loadcurve_data_tensor = torch.from_numpy(loadcurve_data_img)

    batch_loadcurve_img_tensor = batch_generate(loadcurve_data_tensor, batch_size, shuffle) # list
    return batch_loadcurve_img_tensor, data_cnt, max_value, mean_value

def batch_generate(dataset, batch_size, shuffle_or_not):
    dataset_size = dataset.shape[0]
    if shuffle_or_not:
        rand=np.random.permutation(dataset_size)
        dataset = dataset[rand]  # 打乱数据分布
    count = 0
    result = []
    while count * batch_size < dataset_size:
        result.append(dataset[count * batch_size : (count + 1) * batch_size])
        count += 1
    return result
