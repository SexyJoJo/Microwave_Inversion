"""
#厂家模型映射文件
自定义模型映射只需要在下面添加好规则，将自定义的key放到get接口的manufacturer的参数即可
"""
"""
液态水通道索引
自定义模型对于液态水通道也要相应进行配置
"""
T_LIQUID6 = 6  #30.0
T_LIQUID7 = 7  #31.4

# 厂家液态水通道索引配置
    # "AED": 爱尔达厂家,
    # "BFTQ": 北方天穹厂家,
    # "SHLT": 上海雷探厂家,

T_LIQUID_CONFIG = {
    "AED": T_LIQUID7,
    "BFTQ": T_LIQUID6,
    "SHLT": T_LIQUID7,

    "SJQH": T_LIQUID6,
    "22SUO": T_LIQUID7,
    "PINGGU": T_LIQUID7,
}

#lv2输出文件头
HEADER_COLUMNS = [
    'DateTime', '10', '0.000(km)', '0.025(km)', '0.050(km)', '0.075(km)',
    '0.100(km)', '0.125(km)', '0.150(km)', '0.175(km)', '0.200(km)',
    '0.225(km)', '0.250(km)', '0.275(km)', '0.300(km)', '0.325(km)',
    '0.350(km)', '0.375(km)', '0.400(km)', '0.425(km)', '0.450(km)',
    '0.475(km)', '0.500(km)', '0.550(km)', '0.600(km)', '0.650(km)',
    '0.700(km)', '0.750(km)', '0.800(km)', '0.850(km)', '0.900(km)',
    '0.950(km)', '1.000(km)', '1.050(km)', '1.100(km)', '1.150(km)',
    '1.200(km)', '1.250(km)', '1.300(km)', '1.350(km)', '1.400(km)',
    '1.450(km)', '1.500(km)', '1.550(km)', '1.600(km)', '1.650(km)',
    '1.700(km)', '1.750(km)', '1.800(km)', '1.850(km)', '1.900(km)',
    '1.950(km)', '2.000(km)', '2.250(km)', '2.500(km)', '2.750(km)',
    '3.000(km)', '3.250(km)', '3.500(km)', '3.750(km)', '4.000(km)',
    '4.250(km)', '4.500(km)', '4.750(km)', '5.000(km)', '5.250(km)',
    '5.500(km)', '5.750(km)', '6.000(km)', '6.250(km)', '6.500(km)',
    '6.750(km)', '7.000(km)', '7.250(km)', '7.500(km)', '7.750(km)',
    '8.000(km)', '8.250(km)', '8.500(km)', '8.750(km)', '9.000(km)',
    '9.250(km)', '9.500(km)', '9.750(km)', '10.000(km)'
]
