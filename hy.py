#!/usr/bin/env python
# -*- coding:utf-8 -*-

from config import config
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyproj import Proj
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import lightgbm as lgb
import os
import warnings
from utils import geohash_encode

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = '0'


def hashfxn(astring):
    return ord(astring[0])


# 平面坐标转经纬度，供初赛数据使用，复赛可直接忽略
# 标准为西安1980，查询链接：https://mygeodata.cloud/cs2cs/
def transform_xy2lonlat(df):
    x = df['x'].values
    y = df['y'].values
    p = Proj("+proj=tmerc +lat_0=0 +lon_0=120 +k=1 +x_0=500000 +y_0=0 +a=6378140 +b=6356755.288157528 +units=m +no_defs")
    df['lon'], df['lat'] = p(x, y, inverse=True)
    return df


def tfidf(input_values, output_num, output_prefix, seed=1024):
    tfidf_enc = TfidfVectorizer()
    tfidf_vec = tfidf_enc.fit_transform(input_values)
    svd_tmp = TruncatedSVD(n_components=output_num, n_iter=20, random_state=seed)
    svd_tmp = svd_tmp.fit_transform(tfidf_vec)
    svd_tmp = pd.DataFrame(svd_tmp)
    svd_tmp.columns = ['{}_tfidf_{}'.format(output_prefix, i) for i in range(output_num)]
    return svd_tmp


def count2vec(input_values, output_num, output_prefix, seed=1024):
    count_enc = CountVectorizer()
    count_vec = count_enc.fit_transform(input_values)
    svd_tmp = TruncatedSVD(n_components=output_num, n_iter=20, random_state=seed)
    svd_tmp = svd_tmp.fit_transform(count_vec)
    svd_tmp = pd.DataFrame(svd_tmp)
    svd_tmp.columns = ['{}_countvec_{}'.format(output_prefix, i) for i in range(output_num)]
    return svd_tmp


def get_geohash_tfidf(df, group_id, group_target, num):
    df[group_target] = df.apply(lambda x: geohash_encode(x['lat'], x['lon'], 7), axis=1)
    tmp = df.groupby(group_id)[group_target].agg(list).reset_index()
    tmp[group_target] = tmp[group_target].apply(lambda x: ' '.join(x))

    tfidf_tmp = tfidf(tmp[group_target], num, group_target)
    count_tmp = count2vec(tmp[group_target], num, group_target)
    return pd.concat([tmp[[group_id]], tfidf_tmp, count_tmp], axis=1)


def get_grad_tfidf(df, group_id, group_target, num):
    grad_df = df.groupby(group_id)['lat'].apply(lambda x: np.gradient(x)).reset_index()
    grad_df['lon'] = df.groupby(group_id)['lon'].apply(lambda x: np.gradient(x)).reset_index()['lon']
    grad_df['lat'] = grad_df['lat'].apply(lambda x: np.round(x, 4))
    grad_df['lon'] = grad_df['lon'].apply(lambda x: np.round(x, 4))
    grad_df[group_target] = grad_df.apply(
        lambda x: ' '.join(['{}_{}'.format(z[0], z[1]) for z in zip(x['lat'], x['lon'])]), axis=1)

    tfidf_tmp = tfidf(grad_df[group_target], num, group_target)
    return pd.concat([grad_df[[group_id]], tfidf_tmp], axis=1)


def get_sample_tfidf(df, group_id, group_target, num):
    tmp = df.groupby(group_id)['lat_lon'].apply(lambda x: x.sample(frac=0.1, random_state=1)).reset_index()
    del tmp['level_1']
    tmp.columns = [group_id, group_target]
    tmp = tmp.groupby(group_id)[group_target].agg(list).reset_index()
    tmp[group_target] = tmp[group_target].apply(lambda x: ' '.join(x))

    tfidf_tmp = tfidf(tmp[group_target], num, group_target)
    return pd.concat([tmp[[group_id]], tfidf_tmp], axis=1)


# workers设为1可复现训练好的词向量，但速度稍慢，若不考虑复现的话，可对此参数进行调整
def w2v_feat(df, group_id, feat, length):
    print('start word2vec ...')
    data_frame = df.groupby(group_id)[feat].agg(list).reset_index()
    model = Word2Vec(data_frame[feat].values, size=length, window=5, min_count=1, sg=1, hs=1,
                     workers=1, iter=10, seed=1, hashfxn=hashfxn)
    data_frame[feat] = data_frame[feat].apply(lambda x: pd.DataFrame([model[c] for c in x]))
    for m in range(length):
        data_frame['w2v_{}_mean'.format(m)] = data_frame[feat].apply(lambda x: x[m].mean())
    del data_frame[feat]
    return data_frame


def d2v_feat(df, group_id, feat, length):
    print('start doc2vec ...')
    data_frame = df.groupby(group_id)[feat].agg(list).reset_index()
    documents = [TaggedDocument(doc, [i]) for i, doc in zip(data_frame[group_id].values, data_frame[feat])]
    model = Doc2Vec(documents, vector_size=length, window=5, min_count=1, workers=1, seed=1, hashfxn=hashfxn, 
                    epochs=10, sg=1, hs=1)
    doc_df = data_frame[group_id].apply(lambda x: ','.join([str(i) for i in model[x]])).str.split(',', expand=True).apply(pd.to_numeric)
    doc_df.columns = ['{}_d2v_{}'.format(feat, i) for i in range(length)]
    return pd.concat([data_frame[[group_id]], doc_df], axis=1)


def q10(x):
    return x.quantile(0.1)


def q20(x):
    return x.quantile(0.2)


def q30(x):
    return x.quantile(0.3)


def q40(x):
    return x.quantile(0.4)


def q60(x):
    return x.quantile(0.6)


def q70(x):
    return x.quantile(0.7)


def q80(x):
    return x.quantile(0.8)


def q90(x):
    return x.quantile(0.9)


def gen_feat(df):
    df.sort_values(['ID', 'time'], inplace=True)

    df['time'] = df['time'].apply(lambda x: '2019-' + x.split(' ')[0][:2] + '-' + x.split(' ')[0][2:] + ' ' + x.split(' ')[1])
    df['time'] = pd.to_datetime(df['time'])

    df['lat_diff'] = df.groupby('ID')['lat'].diff(1)
    df['lon_diff'] = df.groupby('ID')['lon'].diff(1)
    df['speed_diff'] = df.groupby('ID')['speed'].diff(1)
    df['diff_minutes'] = df.groupby('ID')['time'].diff(1).dt.seconds // 60
    df['anchor'] = df.apply(lambda x: 1 if x['lat_diff'] < 0.01 and x['lon_diff'] < 0.01
                            and x['speed'] < 0.1 and x['diff_minutes'] <= 10 else 0, axis=1)

    lat_lon_neq_zero = df[(df['lat_diff'] != 0) & (df['lon_diff'] != 0)]
    speed_neg_zero = df[df['speed_diff'] != 0]

    df['type'] = df['type'].map({'围网': 0, '刺网': 1, '拖网': 2, 'unknown': -1})
    group_df = df.groupby('ID')['type'].agg({'label': 'mean', 'cnt': 'count'}).reset_index()

    # 获取锚点位置信息
    anchor_df = df.groupby('ID')['anchor'].agg('sum').reset_index()
    anchor_df.columns = ['ID', 'anchor_cnt']

    group_df = group_df.merge(anchor_df, on='ID', how='left')
    group_df['anchor_ratio'] = group_df['anchor_cnt'] / group_df['cnt']

    stat_functions = ['min', 'max', 'mean', 'median', 'nunique', q10, q20, q30, q40, q60, q70, q80, q90]
    stat_ways = ['min', 'max', 'mean', 'median', 'nunique', 'q_10', 'q_20', 'q_30', 'q_40', 'q_60', 'q_70', 'q_80', 'q_90']

    stat_cols = ['lat', 'lon', 'speed', 'direction']
    group_tmp = df.groupby('ID')[stat_cols].agg(stat_functions).reset_index()
    group_tmp.columns = ['ID'] + ['{}_{}'.format(i, j) for i in stat_cols for j in stat_ways]

    lat_lon_neq_group = lat_lon_neq_zero.groupby('ID')[stat_cols].agg(stat_functions).reset_index()
    lat_lon_neq_group.columns = ['ID'] + ['pos_neq_zero_{}_{}'.format(i, j) for i in stat_cols for j in stat_ways]

    speed_neg_zero_group = speed_neg_zero.groupby('ID')[stat_cols].agg(stat_functions).reset_index()
    speed_neg_zero_group.columns = ['ID'] + ['speed_neq_zero_{}_{}'.format(i, j) for i in stat_cols for j in stat_ways]

    group_df = group_df.merge(group_tmp, on='ID', how='left')
    group_df = group_df.merge(lat_lon_neq_group, on='ID', how='left')
    group_df = group_df.merge(speed_neg_zero_group, on='ID', how='left')

    # 获取TOP频次的位置信息，这里选Top3
    mode_df = df.groupby(['ID', 'lat', 'lon'])['time'].agg({'mode_cnt': 'count'}).reset_index()
    mode_df['rank'] = mode_df.groupby('ID')['mode_cnt'].rank(method='first', ascending=False)
    for i in range(1, 4):
        tmp_df = mode_df[mode_df['rank'] == i]
        del tmp_df['rank']
        tmp_df.columns = ['ID', 'rank{}_mode_lat'.format(i), 'rank{}_mode_lon'.format(i), 'rank{}_mode_cnt'.format(i)]
        group_df = group_df.merge(tmp_df, on='ID', how='left')

    tfidf_df = get_geohash_tfidf(df, 'ID', 'lat_lon', 30)
    group_df = group_df.merge(tfidf_df, on='ID', how='left')
    print('geohash tfidf finished.')

    grad_tfidf = get_grad_tfidf(df, 'ID', 'grad', 30)
    group_df = group_df.merge(grad_tfidf, on='ID', how='left')
    print('gradient tfidf finished.')

    sample_tfidf = get_sample_tfidf(df, 'ID', 'sample', 30)
    group_df = group_df.merge(sample_tfidf, on='ID', how='left')
    print('sample tfidf finished.')

    w2v_df = w2v_feat(df, 'ID', 'lat_lon', 30)
    group_df = group_df.merge(w2v_df, on='ID', how='left')
    print('word2vec finished.')
    return group_df


def f1_score_eval(preds, valid_df):
    labels = valid_df.get_label()
    preds = np.argmax(preds.reshape(3, -1), axis=0)
    scores = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'macro_f1_score', scores, True


def build_model(train_, test_, pred, label, cate_cols, split, seed=1024, is_shuffle=True, use_cart=False, get_prob=False):
    n_class = 3
    train_pred = np.zeros((train_.shape[0], n_class))
    test_pred = np.zeros((test_.shape[0], n_class))
    n_splits = 5

    assert split in ['kf', 'skf'], '{} Not Support this type of split way'.format(split)

    if split == 'kf':
        folds = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=seed)
        kf_way = folds.split(train_[pred])
    else:
        folds = StratifiedKFold(n_splits=n_splits, shuffle=is_shuffle, random_state=seed)
        kf_way = folds.split(train_[pred], train_[label])

    print('Use {} features ...'.format(len(pred)))

    params = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'None',
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'num_class': n_class,
        'nthread': 8,
        'verbose': -1,
    }
    for n_fold, (train_idx, valid_idx) in enumerate(kf_way, start=1):
        print('the {} training start ...'.format(n_fold))
        train_x, train_y = train_[pred].iloc[train_idx], train_[label].iloc[train_idx]
        valid_x, valid_y = train_[pred].iloc[valid_idx], train_[label].iloc[valid_idx]

        if use_cart:
            dtrain = lgb.Dataset(train_x, label=train_y, categorical_feature=cate_cols)
            dvalid = lgb.Dataset(valid_x, label=valid_y, categorical_feature=cate_cols)
        else:
            dtrain = lgb.Dataset(train_x, label=train_y)
            dvalid = lgb.Dataset(valid_x, label=valid_y)

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=3000,
            valid_sets=[dvalid],
            early_stopping_rounds=100,
            verbose_eval=100,
            feval=f1_score_eval
        )
        train_pred[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        test_pred += clf.predict(test_[pred], num_iteration=clf.best_iteration)/folds.n_splits
    print(classification_report(train_[label], np.argmax(train_pred, axis=1), digits=4))
    if get_prob:
        sub_probs = ['qyxs_prob_{}'.format(q) for q in ['围网', '刺网', '拖网']]
        prob_df = pd.DataFrame(test_pred, columns=sub_probs)
        prob_df['ID'] = test_['ID'].values
        return prob_df
    else:
        test_['label'] = np.argmax(test_pred, axis=1)
        return test_[['ID', 'label']]


# 不直接对DataFrame做append操作，提升运行速度
def get_data(file_path, model):
    assert model in ['train', 'test'], '{} Not Support this type of file'.format(model)
    paths = os.listdir(file_path)
    tmp = []
    for t in tqdm(range(len(paths))):
        p = paths[t]
        with open('{}/{}'.format(file_path, p), encoding='utf-8') as f:
            next(f)
            for line in f.readlines():
                tmp.append(line.strip().split(','))
    tmp_df = pd.DataFrame(tmp)
    if model == 'train':
        tmp_df.columns = ['ID', 'lat', 'lon', 'speed', 'direction', 'time', 'type']
    else:
        tmp_df['type'] = 'unknown'
        tmp_df.columns = ['ID', 'lat', 'lon', 'speed', 'direction', 'time', 'type']
    tmp_df['lat'] = tmp_df['lat'].astype(float)
    tmp_df['lon'] = tmp_df['lon'].astype(float)
    tmp_df['speed'] = tmp_df['speed'].astype(float)
    tmp_df['direction'] = tmp_df['direction'].astype(int)
    return tmp_df


if __name__ == "__main__":
    TRAIN_PATH = config.train_dir
    TEST_PATH = config.test_dir
    PROB_PATH = config.prob_qyxs
    SAVE_PATH = config.save_path
    USE_PROB = config.use_prob

    train = get_data(TRAIN_PATH, 'train')
    test = get_data(TEST_PATH, 'test')
    train = train.append(test)
    all_df = gen_feat(train)
    del train, test
    use_train = all_df[all_df['label'] != -1]
    use_test = all_df[all_df['label'] == -1]
    use_feats = [c for c in use_train.columns if c not in ['ID', 'label']]
    sub = build_model(use_train, use_test, use_feats, 'label', [], 'kf', is_shuffle=True, use_cart=False,
                      get_prob=USE_PROB)
    if USE_PROB:
        sub.to_csv(PROB_PATH, encoding='utf-8', index=False)
    else:
        sub['label'] = sub['label'].map({0: '围网', 1: '刺网', 2: '拖网'})
        sub.to_csv(SAVE_PATH, encoding='utf-8', header=None, index=False)
