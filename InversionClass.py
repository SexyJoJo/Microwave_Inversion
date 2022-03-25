import pandas as pd
import numpy as np
import re
import json
from fastnumbers import fast_float
from datetime import datetime
import logging
from scipy import ndimage
import copy


class Inversion(object):
    """
    用法：

    初始化类对象:传入 jsonName 初始化 model配置参数

    反演:调用inverting方法完成反演

    inverting需要传入观测设备数据matrix

    """
    __slots__ = ('__json', '__liqRevisalArray', '__revisalArray', '__elements',
                 '__elementId', '__logger', '__compatible', '__brands',
                 '__inputNode', '__inputNode_sunny', '__inputNode_cloudy',
                 '__normaliz_methods', '__inputminmax', '__outputminmax',
                 '__b1', '__w1', '__b2', '__w2', '__normaliz_methods_sunny',
                 '__normaliz_methods_cloudy', '__inputminmax_sunny',
                 '__outputminmax_sunny', '__inputminmax_cloudy',
                 '__outputminmax_cloudy', '__b1_sunny', '__w1_sunny',
                 '__w2_sunny', '__b2_sunny', '__b1_cloudy', '__w1_cloudy',
                 '__w2_cloudy', '__b2_cloudy', '__custom_regparams',
                 '__t_liquid', '__surRevisalArray', '__tempRevisalElement')
    """
        对象变量:
        __json:对应的json模型配置文件
        不同要素共用一组变量，
        __elements: 模型支持的气象要素类型
        __elementId: submodel对应的要素类型
        __liqRevisalArray:水汽订正公式数组  [[[k,b],订正公式]..[[],]] TODO 需求模糊，暂时吧水汽和原来订正的分开写
        __revisalArray:订正公式数组    [[[k,b],订正公式]..[[],]]
        __normaliz_methods:选用的归一化方法，例如极值法等等
        __compatible:模型是否兼容有云无云
        __inputNode:通用模型输入层节点数
        __inputminmax:输入极值矩阵
        __outputminmax:输出极值矩阵
        __b1:输入层-隐藏层b矩阵
        __w1:输入层-隐藏层w矩阵
        __b2:隐藏层-输出层b矩阵
        __w2:隐藏层-输出层w矩阵
        分晴云天的参数
        __inputNode_sunny:晴天模型输入层节点数
        __inputNode_cloudy:云天模型输入层节点数
        __normaliz_methods_sunny:晴天模型归一化方法
        __normaliz_methods_cloudy:云天模型归一化方法
        __inputminmax_sunny:晴天输入极值矩阵
        __outputminmax_sunny:晴天输出极值矩阵
        __inputminmax_cloudy:云天输入极值矩阵
        __outputminmax_cloudy:云天输出极值矩阵
        __b1_sunny:输入层-隐藏层晴天b矩阵
        __w1_sunny:输入层-隐藏层w矩阵
        __w2_sunny:隐藏层-输出层w矩阵
        __b2_sunny:隐藏层-输出层晴天b矩阵
        __b1_cloudy:输入层-隐藏层云天b矩阵
        __w1_cloudy:输入层-隐藏层云天b矩阵
        __w2_cloudy:隐藏层-输出层云天b矩阵
        __b2_cloudy:隐藏层-输出层云天b矩阵
        __surRevisalArray:地表要素订正数组
        __tempRevisalElement：原始（或者订正后）的地表温度和湿度要素矩阵
        """
    def __init__(self, jsonFileName, t_liquid, regparamsPath=""):

        self.__init_json(jsonFileName, regparamsPath)  # 加载json模型配置文件
        self.__logger = self.__set_logger()
        self.__t_liquid = t_liquid

    def __set_logger(self):
        loger = logging.getLogger('inversion')
        loger.setLevel(logging.DEBUG)
        hander1 = logging.StreamHandler()
        hander1.setLevel(logging.INFO)
        hander1.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        loger.addHandler(hander1)
        return loger

    def __revial(self, l, revialArray, num, rows, flag):
        '''
        订正亮温数据的通用函数
        :param l: 反演矩阵
        :param revialArray: 订正数组: [[],[]]
        :param num: 通道亮温数
        :param rows: 数据条数
        :param flag: 非兼容模型，区分晴云天数据的标志，非兼容模型用的晴天数据没有云数据列
        '''
        # print('revialArray', revialArray)
        for i in range(rows):
            if len(revialArray) == 2:
                # 无云，用晴天公式
                if flag or l[i, num + 4] == 0 and l[i, num +
                                                    6] == 0 and l[i, num +
                                                                  8] == 0:

                    def revisals(x, b, k):
                        return eval(revialArray[0][1])

                    k = revialArray[0][0][0]
                    b = revialArray[0][0][1]
                    l[i, :num] = revisals(l[i, :num], b, k)
                else:

                    def revisals(x, b, k):
                        return eval(revialArray[1][1])

                    k = revialArray[1][0][0]
                    b = revialArray[1][0][1]
                    l[i, :num] = revisals(l[i, :num], b, k)
            elif len(revialArray) == 1:

                def revisals(x, b, k):
                    return eval(revialArray[0][1])

                k = revialArray[0][0][0]
                b = revialArray[0][0][1]
                l[i, :num] = revisals(l[i, :num], b, k)

    def __surface_revision(self, l, revialArray, num, rows):
        '''
        地表辅助气象要素订正函数
        :param l: 反演矩阵:[[]]
        :param revialArray: 订正数组: [[[],[]]]
        :param num: 通道亮温数
        :param rows: 数据条数
        '''
        # print('revialArray', revialArray)
        if len(revialArray) == 0:
            # 没有地表温湿压偏差订正的话，则使用源lv1输入来代替0层的反演输入
            self.__tempRevisalElement = np.around(l[:, num:num + 2], 2)
            return

        # 创建订正公式的函数
        def revisals(x, b, k):
            return eval(revialArray[1])

        # 对温度湿度压强，三个要素进行订正
        # 温度是对摄氏度做订正的，订正完再转化成k氏
        # 对每列做订正

        k = revialArray[0][0]
        b = revialArray[0][1]
        l[:, num] -= 273.15  # 先换成摄氏度
        l[:, num:num + 3] = revisals(l[:, num:num + 3], b, k)
        # 保存温度湿度订正的结果用于输出的时候替换0高度层温度湿度数据
        self.__logger.info("换0高度层气象要素前: ", self.__tempRevisalElement)
        self.__tempRevisalElement = np.around(l[:, num:num + 2], 2)
        # print('----订正地表之后')
        # print(self.__tempRevisalElement)
        self.__logger.info("换0高度层气象要素后: ", self.__tempRevisalElement)
        l[:, num] += 273.15  # 换回k氏温度

    # 液态水偏差订正

    def __liquid_revision(self, l, revialArray, num, rows, flag):
        '''
        液态水订正亮温数据函数
        :param l: 反演矩阵:[[]]
        :param revialArray: 订正数组: [[],[]]
        :param num: 通道亮温数
        :param rows: 数据条数
        :param flag: 非兼容模型，区分晴云天数据的标志，非兼容模型用的晴天数据没有云数据列
        '''
        # print('revialArray', revialArray)
        if len(revialArray) == 0:
            return
        for i in range(rows):
            l[i, :num] = np.round(
                l[i, :num] -
                (np.array(l[i][self.__t_liquid]) * revialArray[0][0][0] +
                 revialArray[0][0][1]), 3)

    def __init_json(self, json_path, regParams_path):
        '''
        初始化json中的部分全局变量.

        :param json_path: json文件名
        '''
        with open(json_path, mode='r', encoding='utf-8') as f:
            try:
                self.__json = json.load(f)
            except:
                raise ValueError("解析模型配置文件时出错!".format(json_path))
            # 初始化 区分 兼容/分晴雨天 的变量 compatible(兼容为True)
            self.__compatible = self.__json['compatible']
            self.__elements = self.__json.setdefault(
                "elements", ["温度", "湿度"])  # 以及设置 self.__elements: 训练模型类型的列表

            if regParams_path:
                with open(regParams_path, mode="r",
                          encoding='utf-8') as reg_file:
                    try:
                        self.__custom_regparams = json.load(reg_file)
                    except:
                        self.__custom_regparams = None
            else:
                self.__custom_regparams = None

    def __init_element_model(self, elementName):
        '''
        传入元素参数如:温度,获取对应的模型参数

        :param elementName: 模型名称 例如 温度 ，湿度
        '''
        # 解析订正公式数组
        if 'input_nodes' in self.__json:
            mapping_nodes = self.__json['input_nodes']['mappingNodes']
        else:
            mapping_nodes = None

        for item in ["regParams", "liqParams", "surfaceParams"]:
            if self.__custom_regparams and item in self.__custom_regparams:
                # print("自定义偏差订正文件：", self.__custom_regparams)
                # temp = self.__custom_regparams[item]
                temp = copy.deepcopy(self.__custom_regparams[item])
            elif item in self.__json:
                temp = self.__json[item]
            else:
                temp = []
            # temp =  if self.__custom_regparams[item] else self.__json[item]
            # print(item, temp)
            if len(temp) == 0:
                temp = []
            else:
                # 重组数组
                temp1 = []
                for i_index, i in enumerate(temp):
                    # 如果存在模型输入通道有灵活调整的情况，进行映射处理
                    if mapping_nodes:
                        for j_index, j in enumerate(i['coeffs']):
                            j_temp = []
                            for index, v in enumerate(j):
                                if index in mapping_nodes:
                                    j_temp.append(v)
                            temp[i_index]['coeffs'][j_index] = j_temp
                        temp1.extend([[temp[i_index]['coeffs'], i['apply']]])
                    else:
                        if item == 'surfaceParams':
                            temp1.extend([i["coeffs"], i['apply']])
                        else:
                            temp1.extend([[i["coeffs"], i['apply']]])
                temp = temp1
            if item == 'regParams':
                self.__revisalArray = temp
            elif item == 'liqParams':
                self.__liqRevisalArray = temp
            elif item == 'surfaceParams':
                self.__surRevisalArray = temp
        # 解析亮温有效通道数
        if 'input_nodes' in self.__json:
            self.__brands = self.__json["input_nodes"]["inputBtNodes"]
        else:
            # 通道个数根据用户选择来确定
            self.__brands = self.__json["equipment"]["bands"]
        # 解析温度、湿度、水汽密度的参数模型
        for submodel in self.__json["submodels"]:
            # 模型兼容有云无云，各个要素的模型只有一套:
            if self.__compatible:
                if submodel["elementName"] == elementName:
                    # 网络输入节点数
                    self.__elementId = submodel["elementId"]
                    self.__inputNode = submodel["nodes"][0]
                    self.__normaliz_methods = submodel["normalization"][
                        "methods"]
                    self.__inputminmax = submodel["normalization"]["params"][0]
                    self.__outputminmax = submodel["normalization"]["params"][
                        1]
                    self.__w1 = np.array(submodel["weightMatrices"][0])
                    self.__w2 = np.array(submodel["weightMatrices"][1])
                    self.__b1 = np.array(submodel["biasVectors"][0])
                    self.__b1 = self.__b1.reshape(len(self.__b1), -1)
                    self.__b2 = np.array(submodel["biasVectors"][1])
                    self.__b2 = self.__b2.reshape(len(self.__b2), -1)
            else:  # 模型参数有云无云各有一套:
                if submodel["elementName"] == elementName:
                    if submodel["condition"] == "sunny":
                        self.__elementId = submodel["elementId"]
                        self.__inputNode_sunny = submodel["nodes"][0]
                        self.__normaliz_methods_sunny = submodel[
                            "normalization"]["methods"]
                        self.__inputminmax_sunny = submodel["normalization"][
                            "params"][0]
                        self.__outputminmax_sunny = submodel["normalization"][
                            "params"][1]
                        self.__w1_sunny = np.array(
                            submodel["weightMatrices"][0])
                        self.__w2_sunny = np.array(
                            submodel["weightMatrices"][1])
                        self.__b1_sunny = np.array(submodel["biasVectors"][0])
                        self.__b1_sunny = self.__b1_sunny.reshape(
                            len(self.__b1_sunny), -1)
                        self.__b2_sunny = np.array(submodel["biasVectors"][1])
                        self.__b2_sunny = self.__b2_sunny.reshape(
                            len(self.__b2_sunny), -1)
                    elif submodel["condition"] == "cloudy":
                        self.__elementId = submodel["elementId"]
                        self.__inputNode_cloudy = submodel["nodes"][0]
                        self.__normaliz_methods_cloudy = submodel[
                            "normalization"]["methods"]
                        self.__inputminmax_cloudy = submodel["normalization"][
                            "params"][0]
                        self.__outputminmax_cloudy = submodel["normalization"][
                            "params"][1]
                        self.__w1_cloudy = np.array(
                            submodel["weightMatrices"][0])
                        self.__w2_cloudy = np.array(
                            submodel["weightMatrices"][1])
                        self.__b1_cloudy = np.array(submodel["biasVectors"][0])
                        self.__b1_cloudy = self.__b1_cloudy.reshape(
                            len(self.__b1_cloudy), -1)
                        self.__b2_cloudy = np.array(submodel["biasVectors"][1])
                        self.__b2_cloudy = self.__b2_cloudy.reshape(
                            len(self.__b2_cloudy), -1)

    def __normalization(self, ar2, npList, flag=True):
        '''
         归一化和反归一化两用函数。

        :param ar2:极值矩阵 ar2[0]:极小值矩阵 ar2[1]:极大值矩阵
        :param npList:输入模型参数矩阵或者训练完后的模型参数矩阵
        :param flag:标记是归一化还是反归一化 True是归一化
        :returns:归一化后的结果
        '''
        # ar2 = np.array(ar2)
        # return (2 * (npList - ar2[0]) / (ar2[1] - ar2[0]) -
        #         1) if flag else (0.5 * (npList + 1) * (ar2[1] - ar2[0]) +
        #                          ar2[0])
        '''
         归一化和反归一化两用函数。

        :param ar2:极值矩阵 ar2[0]:极小值矩阵 ar2[1]:极大值矩阵
        :param npList:输入模型参数矩阵或者训练完后的模型参数矩阵
        :param flag:标记是归一化还是反归一化 True是归一化
        :returns:归一化后的结果
        '''

        # 1. 找出极大值不等于极小值的去归一化
        ar2 = np.array(ar2).astype('float')
        tempArray = np.nonzero(ar2[1] - ar2[0])
        # print("模型参数矩阵：",npList)
        # print("归一化数组：",tempArray)
        nor_Array = npList[:, tempArray]
        nor_ar0 = (ar2[0])[tempArray]
        nor_ar1 = (ar2[1])[tempArray]
        nor_Array = (2 * (nor_Array - nor_ar0) / (nor_ar1 - nor_ar0) -
                     1) if flag else (0.5 * (nor_Array + 1) *
                                      (nor_ar1 - nor_ar0) + nor_ar0)
        # 2. 归完重新索引拼起来
        npList[:, tempArray] = nor_Array
        return npList

    # 隐层到输出层不需要带入激活函数 flag =False则是输入到隐层
    def __modelFit(self,
                   l,
                   b1,
                   w1,
                   b2,
                   w2,
                   revialArray,
                   liqRevialArray,
                   surRevialArray,
                   inputminmax,
                   outputminmax,
                   element,
                   is_smoothnes,
                   flag=False):
        '''
        模型反演过程:
          1. 订正亮温
          2. 归一化
          3. 经过神经网络的矩阵运算
          4. 反归一化
        :param l:模型参数
        :param b1:公式的b1的列向量
        :param w1:输入层到隐藏层权重矩阵
        :param b2:公式的b2的列向量
        :param w2:隐藏层到输出层权重矩阵
        :param revialArray:订正公式相关参数的数组 [[[k,b,],订正公式]，...[云天的（可能无）]]
        :param liqRevialArray:水汽订正公式相关参数的数组 [[[k,b,],订正公式]，...[云天的（可能无）]]
        :param surRevialArray:地表辅助气象要素订正公式相关参数的数组 [[k,b,],"订正公式"]
        :param inputminmax:输入的极值矩阵 [[极小值矩阵],[极大值矩阵]]
        :param outputminmax:输出的极值矩阵 [[极小值矩阵],[极大值矩阵]]
        :param element:如果此时反演是温度转换成摄氏度
        :param flag:标记是晴天并且模型数据无云数据，作为额外的判断晴天依据(因为晴天模型 无云参数) True为晴天
        :param is_smoothnes:布尔字段，True为需要平滑False为不需要平滑
        :returns: 单要素 单天气/兼容 的模型输出参数   [各高度要素值]
        '''
        # 水汽通道

        # 1.订正亮温(先订正水汽再订正传统公式)
        # 读出订正公式，订正
        [rows, cols] = l.shape
        num = len(self.__brands)
        self.__surface_revision(l, surRevialArray, num, rows)
        self.__liquid_revision(l, liqRevialArray, num, rows, flag)
        self.__revial(l, revialArray, num, rows, flag)
        # 2.归一化
        l = self.__normalization(inputminmax, l)
        # 3.模型计算
        # 按公式计算带入激活函数
        l = np.dot(w1, l.T) + b1
        l = 2 / (1 + np.exp(-2 * l)) - 1
        # 隐藏层到输出层
        l = np.dot(w2, l) + b2
        # 4.反归一化
        l = self.__normalization(outputminmax, l.T, False)
        if element == '温度':
            l -= 273.15
        # 如果需要光滑
        if is_smoothnes:
            l = self.__smoothnes(l, self.__elementId)
        # TODO
        if self.__elementId in [12, 13]:
            # 湿度和水汽密度两个要素的反演  反演结果中如果出现负值 先全部置为0
            l[l < 0] = 0
            l[l >= 100] = 95
        return np.round(l, decimals=2)

    def __outPutToFile(self, ress):
        '''
        将多个要素的同时间的记录组合在一起

        2020/05/22 13:00      11	温度要素对应

        2020/05/22 13:00      13	湿度要素对应

        :param ress:ress = 各个模型生成的模型参数的数据的列表[[要素一模型生成的数据]，[dateTime, elementId, 各个高度的数据]...]
        :returns:
        2020/05/22 13:00      11	各个高度温度要素对应

        2020/05/22 13:00      13	各个高度湿度要素对应

        2020/05/22 13:02      11	各个高度温度要素对应

        2020/05/22 13:02      13	各个高度湿度要素对应


        matrix :输入lv1矩阵
        '''

        if len(ress) < 1:  # 无
            return np.array([])
        rows, cols = ress[0].shape
        rows += 1

        # 堆叠 吧多要素的整个到一
        # print(ress)
        # 将0高度层的温度湿度替换成订正之后的值或者源lv1输入值（未订正时使用输入）
        # 温度
        ress[0][:, 2] = np.round(self.__tempRevisalElement[:, 0] - 273.16, 2)
        # 湿度
        ress[2][:, 2] = self.__tempRevisalElement[:, 1]

        # for i in range(2):
        #     self.__logger.info("-----换0高度层气象要素前--")
        #     self.__logger.info(ress[i][:, 2])
        #     if i == 0:
        #         # 温度
        #         ress[0][:, 2] = self.__tempRevisalElement[:, i] - 273.16
        #     else:
        #         # 湿度
        #         ress[2][:, 2] = self.__tempRevisalElement[:, i]
        #     self.__logger.info("-----换0高度层气象要素后--")
        #     self.__logger.info(ress[i][:, 2])

        # if len(self.__surRevisalArray) > 0:
        #     for i in range(2):
        #         self.__logger.info("-----换0高度层气象要素前--")
        #         self.__logger.info(ress[i][:, 2])
        #         ress[i][:, 2] = self.__tempRevisalElement[:, i]
        #         self.__logger.info("-----换0高度层气象要素后--")
        #         self.__logger.info(ress[i][:, 2])

        result = np.stack(ress, axis=1)
        g = np.vstack((i for i in result))
        self.__logger.debug(g.shape)

        return g

    def __separation(self, matrix):
        '''
        按照云数据分离晴云天（不兼容模型使用)

        时间列单独分离出来为了训练完之后 按照时间顺序合并回去

        :param matrix: 模型输入参数
        :returns: sun[:, :1]:晴天时间  sun[:, 1:]：晴天模型数据 云天类推
        '''
        num = len(self.__brands)  # 找云数据在哪里
        sunIndex = (matrix[:, num + 5] == 0) & (matrix[:, num + 7] == 0) & (
            matrix[:, num + 9] == 0)
        sun = matrix[sunIndex]
        cloud = matrix[~sunIndex]
        return sun[:, :1], sun[:, 1:].astype(
            np.float), cloud[:, :1], cloud[:, 1:].astype(np.float)

    def __mergeRes(self, tempDate, tempres, flag=True):
        '''
        按照时间顺序吧不同模型训练出来的结果合并
        :param tempDate: 时间列 [[晴天时间],[云天时间]]
        :param tempres: 训练完后的模型结果  [[晴天模型结果],[云天模型结果]]
        :param flag: 标记是兼容模型还是不兼容 True是不兼容模型
        :returns:时间+模型结果 的列表

        例子:

        2020/05/22 13:00      11	各个高度某要素对应

        2020/05/22 13:02      12	各个高度某要素对应

        '''
        l = None
        if flag:  # 吧时间列合并进去，排序
            for i in range(2):
                tempres[i] = np.insert(tempres[i], [0], tempDate[i], axis=1)
            l = np.concatenate((tempres[0], tempres[1]), axis=0)
            l = l[l[:, 0].argsort()]
        else:
            l = np.insert(tempres, [0], tempDate, axis=1)
        return l

    def __smoothnes(self, data, type):
        '''
        平滑函数
        :param data: 反演后的数据
        :param type: 要素类别: 11:温度 12:相对湿度 13：水汽密度
        :returns: 处理后的ndarray
        '''
        k = np.ones((5, 5), dtype='float32') / 25
        data = ndimage.convolve(data, k, mode='nearest', origin=0)
        if type == 12:  # 相对湿度如果平滑后小于0则平滑为0
            data[data <= 0] = 0
        elif type == 13:  # 水汽密度如果平滑后小于5则平滑为5 大于等于95则平滑为95
            data[data <= 5] = 5
            data[data >= 95] = 95
        return np.round(data, decimals=2)

    def inverting(self, matrix, is_smoothnes):
        '''
        反演的主体程序,传入气象观测数据，返回模型反演结果
        :param matrix: [[时间1,有效通道亮温值.....，辅助气象要素1，辅助气象要素2，辅助气象要素3, 云1底,云1厚度,云2底,云2厚度,云3底,云3厚度]
                        ...
                        [时间n,有效通道亮温值.....，辅助气象要素1，辅助气象要素2，辅助气象要素3, 云1底,云1厚度,云2底,云2厚度,云3底,云3厚度]
                        ]
        :param is_smoothnes:布尔字段，True为需要平滑False为不需要平滑
        :returns 返回一个ndarray 表示模型反演结果\n
        返回值例子:

        2020/05/22 13:00      11	各个高度温度要素对应

        2020/05/22 13:00      12	各个高度湿度要素对应

        2020/05/22 13:02      11	各个高度温度要素对应

        2020/05/22 13:02      12	各个高度湿度要素对应
        '''
        ress = []
        elements = self.__elements
        tempData = [None] * 2
        tempDate = [None] * 2
        # 需求有变，标志气象要素的是 温度 11 湿度12 压强13 json里的是1，2,3 在用之前+10 (已废弃，在模型配置文件里直接修改)
        if self.__compatible:
            matrDate = matrix[:, :1]  # 时间列
            matrix = matrix[:, 1:].astype(np.float)  # 去掉时间列
            # self.__logger.info('训练兼容模型')
            # 1.温度
            # 训练各个要素
            for i in range(len(elements)):
                self.__init_element_model(elements[i])
                res = self.__modelFit(
                    matrix.copy(), self.__b1, self.__w1, self.__b2, self.__w2,
                    self.__revisalArray, self.__liqRevisalArray,
                    self.__surRevisalArray, self.__inputminmax,
                    self.__outputminmax, elements[i], is_smoothnes)

                res = np.insert(res, 0, self.__elementId,
                                axis=1).astype(np.str)
                ress.append(self.__mergeRes(matrDate, res, False))
        else:  # 非兼容要分晴云天
            # 1.先把原始数据拆成，晴天数据和云天数据，保留她们的时间列用作合并
            tempDate[0], tempData[0], tempDate[1], tempData[
                1] = self.__separation(matrix)
            self.__logger.info('训练分晴云天模型')
            # 2.反演
            # 晴云天分开来合并
            for i in range(len(elements)):
                self.__init_element_model(elements[i])
                sunRes, CloudyRes = None, None
                l = tempData[0][:, :len(self.__brands) + 3]  # 晴天模型把云数据去掉
                sun_res = self.__modelFit(
                    l.copy(), self.__b1_sunny, self.__w1_sunny,
                    self.__b2_sunny, self.__w2_sunny, self.__revisalArray,
                    self.__liqRevisalArray, self.__surRevisalArray,
                    self.__inputminmax_sunny, self.__outputminmax_sunny,
                    elements[i], is_smoothnes, True)
                sun_res = np.insert(sun_res, 0, self.__elementId,
                                    axis=1).astype(np.str)
                cloudy_res = self.__modelFit(
                    tempData[1], self.__b1_cloudy, self.__w1_cloudy,
                    self.__b2_cloudy, self.__revisalArray,
                    self.__liqRevisalArray, self.__surRevisalArray,
                    self.__inputminmax_cloudy, self.__outputminmax_cloudy,
                    elements[i], is_smoothnes)
                cloudy_res = np.insert(cloudy_res, 0, self.__elementId,
                                       axis=1).astype(np.str)
                ress.append(self.__mergeRes(tempDate, [sun_res, cloudy_res]))
        # 返回结果
        # return self.__outPutToFile(ress)
        return self.__outPutToFile(ress)
