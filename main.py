from InversionClass import Inversion
from inversionAdapter import InversionAPIAdapter
import consts
import pandas as pd
from datetime import datetime
import json
import sys
import numpy as np

# result = {"header": consts.HEADER_COLUMNS, "data": None, "deviceInfo": None}
result = {"header": consts.HEADER_COLUMNS, "data": None}
"""
模块调用入口
input：dict类型
"""


def main(input):
    inversionAPIAdapter = InversionAPIAdapter(input)
    # 获取数据
    lv1Data, cloudData, t_liquid, model_file_name, deviceInfo, corr_config_path = inversionAPIAdapter.parseInput(
    )
    # 数据标准化
    nplist = inversionAPIAdapter.inputPro(lv1Data, cloudData)
    # 进行反演
    output = Inversion(model_file_name,
                       t_liquid=t_liquid,
                       regparamsPath=corr_config_path).inverting(
                           nplist, is_smoothnes=True)
    # 添加数据头
    df = pd.DataFrame(output, columns=consts.HEADER_COLUMNS)
    df["10"] = df["10"].apply(lambda c: c.strip()[0:2])
    # 构建结果
    # result["data"] = np.array(df).tolist()
    result = df
    # if (deviceInfo):
    #     result["deviceInfo"] = deviceInfo
    return result


# if __name__ == '__main__':
#     output = main(test_input)
#     print(output)
