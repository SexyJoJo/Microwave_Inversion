import os
import pandas as pd
import cal_cloud
from main import main


def inverse_one(columns, lv1_row, model_path, manufacturer, station, sounding_dir, corr_path=''):
    sounding_file = cal_cloud.find_recent_sounding(lv1_time=lv1_row[0], station=station, sounding_dir=sounding_dir)
    try:
        cloud_nodes = cal_cloud.compute_cloud(sounding_file)
    except FileNotFoundError:
        cloud_nodes = [0, 0, 0, 0, 0, 0]
    print(lv1_row[0], cloud_nodes)
    test_input = {
        'lv1': [
            columns,
            lv1_row,
        ],
        'cloud': [
            ['Time', 'Bottom1', 'Thick1', 'Bottom2', 'Thick2', 'Bottom3', 'Thick3'],
            [lv1_row[0]] + cloud_nodes,
        ],
        'manufacturer': manufacturer,
        "model_config_path": model_path,
        "corr_config_path": corr_path
    }

    output = main(test_input)
    return output


def inverse_one_station(model_path, lv1_dir, sounding_dir, out_dir, manufacturer):
    """
    将lv1_dir目录中的所有lv1文件反演为lv2
    """
    for root, _, files in os.walk(lv1_dir):
        for file in files:
            if file.endswith('RAW_D.txt'):
                station = file.split('_')[3]
                print(f'正在反演{file}')
                lv1 = pd.read_csv(os.path.join(root, file), skiprows=2, encoding='gbk')
                lv1 = lv1.drop(['Record', 'QCFlag', 'Az(deg)', 'El(deg)', 'QCFlag_BT'], axis=1)
                lv1.rename(columns={'DateTime': 'Date/Time'}, inplace=True)

                lv2_df = pd.DataFrame()
                for index, row in lv1.iterrows():
                    lv2_line = inverse_one(lv1.columns, row, model_path, manufacturer, station, sounding_dir)
                    lv2_line.insert(0, 'Record', index + 1)
                    lv2_line.insert(3, 'SurTem(℃)', row['SurTem(℃)'])
                    lv2_line.insert(4, 'SurHum(%)', row['SurHum(%)'])
                    lv2_line.insert(5, 'SurPre(hPa)', row['SurPre(hPa)'])
                    lv2_line.insert(6, 'Tir(℃)', row['Tir(℃)'])
                    lv2_line.insert(7, 'Rain', row['Rain'])
                    lv2_df = pd.concat([lv2_df, lv2_line], axis=0)
                print(lv2_df)

                file_ele = file.split('_')
                file_ele[5] = 'P'
                file_ele[-2] = 'CP'
                lv2_name = '_'.join(file_ele)
                out_path = os.path.join(out_dir, station)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                lv2_df.to_csv(os.path.join(out_path, lv2_name), index=False, encoding='utf-8')


if __name__ == '__main__':
    inverse_one_station(
        model_path=r'./modelConfig/廊坊18-21EC_2af00ce4-a203-11ec-8899-e8f408c7c544.json',
        lv1_dir=r'D:\Data\microwave radiometer\Measured brightness temperature\54510廊坊',
        sounding_dir=r'D:\Data\microwave radiometer\Sounding',
        out_dir=r'D:\Data\microwave radiometer\LV2',
        manufacturer='AED'
    )
