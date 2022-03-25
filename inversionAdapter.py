"""
反演程序API接口适配器
反演-API接口程序，负责抓取数据，反演预处理和上传数据
"""
__auther__ = 'lnp'

import pandas as pd
from datetime import datetime
import json
import consts
import sys
import traceback

EXIT = sys.exit


class InversionAPIAdapter(object):
    """
    初始化参数:
        __input: "输入(dict)",
    """

    __slots__ = ("__input")

    def __init__(self, input):
        self.__input = input

    # 生成标准输入层数据
    def inputPro(self, datas, clouds):
        """
            datas：亮温数据列表(1+3+n) 
            cloud:云数据列表:(1+6)
            1.读入两张表数据
            2.吧两张表的时间转化为同一格式高
            3.按时间列内连接两张表格（需要做缺值处理）
            4.分晴云天订正
            5.归一化
        """
        df = pd.DataFrame(datas, columns=datas[0])
        #lv数据时间列表
        dateTime = df.iloc[1:, 0]
        #时间以及温室压三要素
        elsments = df.iloc[1:, 1:4]
        #亮温数据
        data = df.iloc[1:, 6:]
        df1 = pd.concat([dateTime, data, elsments], axis=1)

        df1['SurTem(℃)'] += 273.15  # 单位转换
        # df1['Tir(C)'] += 273.15  # 单位转换

        # 三层云转换为单层云
        # for index, cloud in enumerate(clouds):
        #     if index == 0:
        #         cloud[1:] = ['BottomHeight', 'TopHeight']
        #     else:
        #         cloud[1:] = self.calculateCloudIntervalHeight(cloud[1:])

        #此处的云数据有效列已经处理过了
        df2 = pd.DataFrame(clouds[1:], columns=clouds[0])
        df2.replace("-", 0.0, inplace=True)
        # 2.处理合并两张表的数据
        """
        1.吧两张表的时间转化为同一格式
        2.云数据表格处理为 底，高
        3.内连接两张表格（需要做缺值处理）
        """
        # 以时间合并所以要转化成统一格式
        df1['Date/Time'] = df1.iloc[:, 0].apply(lambda s: datetime.strptime(
            s, "%Y-%m-%d %H:%M:%S").strftime("%Y/%m/%d %H:%M")).values
        df2['Time'] = df2.iloc[:, 0].apply(lambda s: datetime.strptime(
            s, "%Y-%m-%d %H:%M:%S").strftime("%Y/%m/%d %H:%M")).values
        df2.rename(columns={'Time': 'Date/Time'}, inplace=True)
        # 按时间整合
        df3 = pd.merge(df1, df2, on='Date/Time')
        # 吧pandas表格结构的value提出来便于后面ndarray操作
        nplist = df3.values
        return nplist

    # 解析输入
    def parseInput(self):
        try:
            lv1Data = self.__input['lv1']
            cloudData = self.__input['cloud']
            # 初始化液态水通道 14\16->7; 22->6
            if self.__input['manufacturer'] is not None and self.__input[
                    'manufacturer'] != "":
                t_liquid = consts.T_LIQUID_CONFIG[self.__input['manufacturer']]
            else:
                t_liquid = 7
            if "deviceInfo" in self.__input:
                deviceInfo = self.__input['deviceInfo']
            else:
                deviceInfo = None

            #默认模型文件
            # model_file_name = consts.MANU_CONFIG_MAPPING[self.__input['manufacturer']]
            # 自定义设置模型文件路径
            model_config_path = self.__input['model_config_path']
            if model_config_path is not None and model_config_path != "":
                model_file_name = model_config_path
            else:
                raise Exception("未传入模型文件")

            # 自定义偏差订正文件路径
            corr_config_path = self.__input['corr_config_path']
            if corr_config_path is not None and corr_config_path != "":
                pass
            else:
                corr_config_path = ""
            return lv1Data, cloudData, t_liquid, model_file_name, deviceInfo, corr_config_path
        except Exception as e:
            print("解析输入出错")
            print(traceback.format_exc())
            EXIT(1)

    """
    根据6个云节点数据计算出2个最大区间节点：云底高度、云顶高度
    """

    def calculateCloudIntervalHeight(self, cloud_List):
        cloud_bottom = 0
        cloud_top = 0
        i = 0
        while i < len(cloud_List):
            if cloud_List[i] and not cloud_bottom:
                cloud_bottom = cloud_List[i]
            i += 2

        cloud_top = cloud_bottom + cloud_List[1] + cloud_List[3] + cloud_List[5]
        cloud_top = 10000 if cloud_top > 10000 else cloud_top

        return [cloud_bottom, cloud_top]