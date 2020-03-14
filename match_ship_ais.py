#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:juzphy
# datetime:2020/3/14 11:07 下午
from utils import geohash_encode
import os
import pandas as pd
from tqdm import tqdm


def match_ship_in_ais(train_file, ais_file, use_lat_lon=False):
    paths = os.listdir(ais_file)
    match_df = pd.DataFrame()

    # 复赛训练集合并文件
    train = pd.read_csv(train_file)
    train.columns = ['ID', 'lat', 'lon', 'speed', 'direction', 'time', 'type']
    print('origin train records num:', train.shape[0])
    train['time'] = train['time'].apply(lambda x: x[:7])
    train = train.drop_duplicates(subset=['ID', 'time', 'lat', 'lon'])

    print('after drop duplicates, train records num:', train.shape[0])
    if not use_lat_lon:
        train['geohash7'] = train.apply(lambda x: geohash_encode(x['lat'], x['lon'], 7), axis=1)
    else:
        train['lat'] = train['lat'].round(3)
        train['lon'] = train['lon'].round(3)
    max_lat, min_lat, max_lon, min_lon = train['lat'].max(), train['lat'].min(), train['lon'].max(), train['lon'].min()
    print('*'*100)
    for t in tqdm(range(len(paths))):
        p = paths[t]
        print(f'start match train ship with ais: {p}')
        ais = pd.read_csv(f'{ais_file}/{p}')
        ais.columns = ['ais_speed', 'ais_direction', 'ais_id', 'time', 'ais_lon', 'ais_lat']
        print('origin ais records num:', ais.shape[0])
        ais['time'] = ais['time'].apply(lambda x: x[:7])
        ais = ais.drop_duplicates(subset=['ais_id', 'time', 'ais_lat', 'ais_lon'])
        print('after drop duplicates, ais records num:', ais.shape[0])
        ais = ais[(ais['ais_lat']>min_lat) & (ais['ais_lat']<max_lat) & (ais['ais_lon']>min_lon) & (ais['ais_lon']<max_lon)]
        print('remove beyond threshold lat and lon, ais records num:', ais.shape[0])
        if not use_lat_lon:
            ais['geohash7'] = ais.apply(lambda x: geohash_encode(x['ais_lat'], x['ais_lon'], 7), axis=1)
            merge_df = train.merge(ais, how='inner', on=['time', 'geohash7'])
        else:
            ais['ais_lat'] = ais['ais_lat'].round(3)
            ais['ais_lon'] = ais['ais_lon'].round(3)
            merge_df = train.merge(ais, how='inner', left_on=['time', 'lat', 'lon'], right_on=['time', 'ais_lat', 'ais_lon'])
        merge_df['cnt'] = merge_df.groupby('ID')['ais_id'].transform('nunique')
        merge_df = merge_df[merge_df['cnt'] == 1]
        del merge_df['cnt']
        print('now match records num:', merge_df['ID'].nunique())
        match_df = match_df.append(merge_df)
        print('total match records num:', match_df['ID'].nunique())
        print('#'*100)
    return match_df


if __name__ == "__main__":
    save_path = 'match.csv'
    match_result = match_ship_in_ais('train_round2.csv', 'round2_ais_20200310')
    match_result.to_csv(save_path, index=False)
