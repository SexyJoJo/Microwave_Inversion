# coding=UTF-8

import pandas as pd
from math import log10
from datetime import datetime, timedelta
import os
# from webapi.settings import rsfile_folder


# 根据高度分类计算入云判定的湿度阈值
def calculate_rh(height):
    height_km = height / 1000
    if 0 <= height < 1000:
        rh = 91
    elif 1000 <= height < 2000:
        rh = -6.416 * height_km + 97
    elif 2000 <= height < 7562:
        rh = -1.223 * height_km + 87
    elif 7562 <= height < 10000:
        rh = -4 * height_km + 108
    else:
        rh = 68

    return rh


# 由每层的高度、湿度计算冰面饱和水汽压下的空气相对湿度
def calculate_humidity(temperature, humidity):
    T = temperature
    if T < 273.15:
        logEw = 10.79574 * (1 - 273.16 / T) - 5.028 * log10(T / 273.16) + \
                1.50475 * 0.0001 * (1 - 10 ** (-8.2969 * (T / 273.16 - 1))) + \
                0.42873 * 0.001 * (10 ** (4.76955 * (1 - 273.16 / T)) - 1) + 0.78614
        logEi = -9.09685 * (273.16 / T - 1) - 3.56654 * log10(273.16 / T) + 0.87682 * (1 - T / 273.16) + 0.78614
        Ur = humidity * ((10 ** logEw) / (10 ** logEi))
        humidity = Ur
    else:
        pass
    return humidity


# 根据得到的湿度，判断是否入云，计算云层信息
def calculate_cloud(height_temperature_humidity_list):
    humi_list = []
    rh_list = []
    for data in height_temperature_humidity_list:
        new_humidity = calculate_humidity(data[1], data[2])
        humi_list.append(new_humidity)
        rh_list.append(calculate_rh(data[0]))

    humi_list[3] = 99
    cloud_bottom = []
    cloud_top = []
    bottom_index = []
    top_index = []
    for index, humidity in enumerate(humi_list):

        if humidity >= rh_list[index] and humi_list[index - 1] < rh_list[index - 1]:
            bottom = height_temperature_humidity_list[index][0]
            cloud_bottom.append(bottom)
            bottom_index.append(index)
        if len(cloud_bottom) > len(cloud_top):
            if height_temperature_humidity_list[index][0] == height_temperature_humidity_list[-1][0]:
                cloud_top.append(height_temperature_humidity_list[-1][0])
                top_index.append(index)
                continue

            if humidity < rh_list[index] and humi_list[index - 1] >= rh_list[index - 1]:
                top = height_temperature_humidity_list[index][0]
                if top >= 500:
                    cloud_top.append(top)
                    top_index.append(index)
                else:
                    cloud_bottom.pop()
                    bottom_index.pop()

    cloud_data = []
    for i in range(0, len(cloud_top)):

        cloud_data.append([bottom_index[i], cloud_bottom[i], top_index[i], cloud_top[i]])

    todo_remove_list = []
    for one_data in cloud_data:
        # 云层判断
        if one_data[3] < one_data[1] and one_data[2] > one_data[0]:
            todo_remove_list.append(one_data)
            continue
        if one_data[3] - one_data[1] < 80:

            for i in range(one_data[0], one_data[2] + 1, 1):
                if humi_list[i] < rh_list[i] + 3:
                    yc_flag = True
                else:
                    yc_flag = False
            if yc_flag:
                todo_remove_list.append(one_data)

    for one_data in todo_remove_list:
        cloud_data.remove(one_data)

    new_cloud_data = []

    yjc_flag = False
    x = 0
    for j in range(0, len(cloud_data), 1):
        if yjc_flag is False:
            new_cloud_data.append(cloud_data[j])
        # 云夹层判断
        if j == len(cloud_data) - 1:
            break
        if cloud_data[j + 1][1] - new_cloud_data[x][3] < 300 and cloud_data[j + 1][1] > new_cloud_data[x][3]:

            for i in range(new_cloud_data[x][2] + 1, cloud_data[j + 1][0], 1):
                # if humi_list[i] > rh_list[i] - 5:
                if humi_list[i] > 5:
                    yjc_flag = True
                else:
                    yjc_flag = False
            if yjc_flag:

                new_cloud_data[x][2] = cloud_data[j + 1][2]
                new_cloud_data[x][3] = cloud_data[j + 1][3]
            else:
                x = x + 1
        else:
            yjc_flag = False
            x = x + 1

    return new_cloud_data


def convert_numberic(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')


# 云层计算 入口函数
def compute_cloud(filename):
    # # 判断文件是新格式还是旧格式
    # if rsfile_folder in filename:
    #     df = pd.read_csv(filename, sep=' ', skiprows=6, header=None, nrows=3000)
    #     convert_numberic(df, [1, 2, 3, 11])
    #     df.interpolate(inplace=True)
    #
    #     hei_temp_humi_list = []
    #     for i in range(0, len(df), 1):
    #         if df.iat[i, 11] - df.iat[0, 11] > 10000:
    #             break
    #         hei_temp_humi_list.append([float(df.iat[i, 11]), float(df.iat[i, 1]) + 273.15, float(df.iat[i, 3])])
    # else:
    df = pd.read_csv(filename, sep=' ', skiprows=0, header=None)
    convert_numberic(df, [0, 1, 2, 3])

    hei_temp_humi_list = []
    for i in range(0, len(df), 1):
        if df.iat[i, 3] - df.iat[0, 3] > 10000:
            break
        hei_temp_humi_list.append([float(df.iat[i, 3]), float(df.iat[i, 0]) + 273.15, float(df.iat[i, 2])])

    cloud_info = calculate_cloud(hei_temp_humi_list)
    if len(cloud_info) == 0:
        result = [0, 0, 0, 0, 0, 0]
    elif len(cloud_info) == 1:
        result = [cloud_info[0][1], cloud_info[0][3] - cloud_info[0][1], 0, 0, 0, 0]
    elif len(cloud_info) == 2:
        result = [cloud_info[0][1], cloud_info[0][3] - cloud_info[0][1],
                  cloud_info[1][1], cloud_info[1][3] - cloud_info[1][1], 0, 0]
    elif len(cloud_info) == 3:
        result = [cloud_info[0][1], cloud_info[0][3] - cloud_info[0][1],
                  cloud_info[1][1], cloud_info[1][3] - cloud_info[1][1],
                  cloud_info[2][1], cloud_info[2][3] - cloud_info[2][1]]
    else:
        result = [0, 0, 0, 0, 0, 0]
    return result

def find_recent_sounding(lv1_time, station, sounding_dir):
    """寻找距离观测时间最近的sounding文件"""
    obs_time = datetime.strptime(lv1_time, "%Y-%m-%d %H:%M:%S")
    if 2 <= obs_time.hour < 14:
        sounding_time = lv1_time[:4] + lv1_time[5:7] + lv1_time[8:10] + '080000'
        world_time = datetime.strptime(sounding_time, '%Y%m%d%H%M%S') + timedelta(hours=8)
        world_time = datetime.strftime(world_time, '%Y%m%d%H%M%S')
    elif 14 <= obs_time.hour <= 23:
        sounding_time = lv1_time[:4] + lv1_time[5:7] + lv1_time[8:10] + '200000'
        world_time = datetime.strptime(sounding_time, '%Y%m%d%H%M%S') + timedelta(hours=8)
        world_time = datetime.strftime(world_time, '%Y%m%d%H%M%S')
    else:
        temp_time = datetime.strftime((obs_time - timedelta(days=1)), "%Y-%m-%d %H:%M:%S")
        sounding_time = temp_time[:4] + temp_time[5:7] + temp_time[8:10] + '200000'
        world_time = datetime.strptime(sounding_time, '%Y%m%d%H%M%S') + timedelta(hours=8)
        world_time = datetime.strftime(world_time, '%Y%m%d%H%M%S')

    filename = station + '_' + world_time + '.txt'
    print(filename)
    full_path = os.path.join(sounding_dir, station, world_time[:4], world_time[4:6], filename)
    return full_path

