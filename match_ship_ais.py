#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:juzphy
# datetime:2020/3/14 11:07 下午
from utils import geohash_encode
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def match_ship_in_ais(train_file, ais_file, threshold, cnt_limit=50):
    paths = os.listdir(ais_file)
    match_df = pd.DataFrame()

    # 复赛训练集
    train = pd.read_csv(train_file)
    train.columns = ['ID', 'lat', 'lon', 'speed', 'direction', 'time', 'type']
    train.sort_values(['ID', 'time'], inplace=True)
    train['lat'] = train['lat'].round(2)
    train['lon'] = train['lon'].round(2)
    print('origin train records num:', train.shape[0])
    train['time'] = train['time'].apply(lambda x:x[:4])
    train = train.drop_duplicates(subset=['ID', 'time', 'lat', 'lon'])
    print('after drop duplicates, train records num:', train.shape[0])

    max_lat, min_lat, max_lon, min_lon = train['lat'].max(), train['lat'].min(), train['lon'].max(), train['lon'].min()
    print('*'*100)
    train_agg = train.groupby('ID')['lat', 'lon'].agg(list).reset_index()
    train_cnt = train.groupby(['ID'])['time'].agg({'cnt': 'count'}).reset_index()
    train_cnt = train_cnt[train_cnt['cnt'] > cnt_limit]
    train_agg = train_agg[train_agg['ID'].isin(train_cnt['ID'].unique())]
    train_time = train.groupby('ID')['time', 'type'].agg('min').reset_index()
    train_agg = train_agg.merge(train_time, on='ID', how='left')
    train_agg['lon_lat'] = train_agg.apply(lambda x: {'{}_{}'.format(z[0], z[1]) for z in zip(x['lat'], x['lon'])}, axis=1)
    del train_agg['lat'], train_agg['lon']
    for t in tqdm(range(len(paths))):
        p = paths[t]
        print(f'start match train ship with ais: {p}')
        ais = pd.read_csv(f'{ais_file}/{p}')
        ais.columns = ['ais_speed', 'ais_direction', 'ais_id', 'time', 'ais_lon', 'ais_lat']
        ais.sort_values(['ais_id', 'time'], inplace=True)
        print('origin ais records num:', ais.shape[0])
        ais['time'] = ais['time'].apply(lambda x:x[:4])
        ais['ais_lat'] = ais['ais_lat'].round(2)
        ais['ais_lon'] = ais['ais_lon'].round(2)
        ais = ais.drop_duplicates(subset=['ais_id', 'time', 'ais_lat', 'ais_lon'])
        print('after drop duplicates, ais records num:', ais.shape[0])
        ais = ais[(ais['ais_lat'] > min_lat) & (ais['ais_lat'] < max_lat) & (ais['ais_lon'] > min_lon) & (ais['ais_lon'] < max_lon)]
        print('remove beyond threshold lat and lon, ais records num:', ais.shape[0])
        ais_cnt = ais.groupby(['ais_id'])['time'].agg({'cnt': 'count'}).reset_index()
        ais_cnt = ais_cnt[ais_cnt['cnt'] > cnt_limit]
        ais_time = ais.groupby('ais_id')['time'].agg('min').reset_index()
        ais_agg = ais.groupby('ais_id')['ais_lat', 'ais_lon'].agg(list).reset_index()
        ais_agg = ais_agg[ais_agg['ais_id'].isin(ais_cnt['ais_id'].unique())]
        ais_agg = ais_agg.merge(ais_time, on='ais_id', how='left')
        ais_agg['ais_lon_lat'] = ais_agg.apply(lambda x: {'{}_{}'.format(z[0], z[1]) for z in zip(x['ais_lat'], x['ais_lon'])}, axis=1)
        del ais_agg['ais_lat'], ais_agg['ais_lon']
        all_df = train_agg.merge(ais_agg, on='time', how='inner')
        if all_df.shape[0] == 0:
            continue
        all_df['share_size'] = all_df.apply(lambda x: len(x['lon_lat'] & x['ais_lon_lat']), axis=1)
        all_df['ratio_x'] = all_df.apply(lambda x: x['share_size']/len(x['ais_lon_lat']), axis=1)
        all_df['ratio_y'] = all_df.apply(lambda x: x['share_size']/len(x['lon_lat']), axis=1)
        all_df = all_df[(all_df['ratio_x'] >= threshold) & (all_df['ratio_y'] >= threshold)]
        all_df['ais_file_name'] = p.split('.')[0]
        match_df = match_df.append(all_df[['ID', 'type', 'ais_id', 'share_size', 'ratio_x', 'ratio_y', 'ais_file_name']])
        print('now match:', match_df.shape[0])
        print('*'*100)
    return match_df


# 可视化匹配成功的数据，再做筛选
def compare_plot(ids, ais_file):
    map_net = {'刺网': 'cw', '围网': 'ww', '拖网': 'tw'}
    fig = plt.figure(figsize=(8, 4))
    plt.subplot(2, 2, 1)
    train = pd.read_csv('train_round2.csv')
    left = train[train['渔船ID'] == ids[0]]
    left_type = left['type'].map(map_net).min()
    plt.scatter(left['lat'], left['lon'], color='blue')
    ais = pd.read_csv(f'{ais_file}/{ids[2]}.csv')
    ais = ais[ais['ais_id'] == ids[1]]
    plt.subplot(2, 2, 2)
    plt.scatter(ais['lat'], ais['lon'], color='red')
    fig.suptitle(f'match ID:{ids[0]}, type: {left_type}, ais_id:{ids[1]}, ais_file:{ids[2]}')
    plt.show()


if __name__ == "__main__":
    save_path = 'match.csv'
    ais_path = 'round2_ais_20200310'
    match_result = match_ship_in_ais('train_round2.csv', ais_path, 0.5, cnt_limit=50)
    match_result.to_csv(save_path, index=False)
    for m in match_result[['ID', 'ais_id', 'ais_file_name']].values:
        compare_plot(m, ais_path)
