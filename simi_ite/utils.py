import numpy as np
import sklearn
from copy import deepcopy
from auuc import auuc_score
import ot
import torch

class StandardScaler:

    # We provide our DIY scaler operator since the treatment column is special
    def __init__(self):
        self.mean = 0
        self.std = 0

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-6
        self.mean[-1] = 0  # Do NOT scale the treatment column
        self.std[-1] = 1
        # self.mean[-1] = 0  # Do NOT scale the counterfactual outcome column (it is just used in evaluation)
        # self.std[-1] = 1

        # self.mean = np.zeros_like(self.mean)
        # self.std = np.ones_like(self.mean)

    def transform(self, data):
        data = (data - self.mean) / (self.std)
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def reverse_y(self, yf):
        y = yf * self.std[-2] + self.mean[-2]
        return y


def metric_update(metric: dict(), metric_: dict(), epoch) -> dict():
    """
    Update the metric dict
    :param metric: self.metric in the class Estimator, each value is array
    :param metric_: output of metric() function, each value is float
    :return:
    """
    for key in metric_.keys():
        metric[key] = np.concatenate([metric[key], [metric_[key]]])
    info = "Epoch {:>3}".format(epoch)
    return metric


def metric_export(path, train_metric, eval_metric, test_metric):

    with open(path+'/run.txt', 'w') as f:
        f.write("mode,pehe,auuc,rauuc,ate,att,r2_f,r2_cf,rmse_f,rmse_cf\n")
        f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
            'train',
            train_metric['pehe'],
            train_metric['auuc'],
            train_metric['rauuc'],
            train_metric['mae_ate'],
            train_metric['mae_att'],
            train_metric['r2_f'],
            train_metric['r2_cf'],
            train_metric['rmse_f'],
            train_metric['rmse_cf']
        ))
        f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
            'eval',
            eval_metric['pehe'],
            eval_metric['auuc'],
            eval_metric['rauuc'],
            eval_metric['mae_ate'],
            eval_metric['mae_att'],
            eval_metric['r2_f'],
            eval_metric['r2_cf'],
            eval_metric['rmse_f'],
            eval_metric['rmse_cf']
        ))

        f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
            'test',
            test_metric['pehe'],
            test_metric['auuc'],
            test_metric['rauuc'],
            test_metric['mae_ate'],
            test_metric['mae_att'],
            test_metric['r2_f'],
            test_metric['r2_cf'],
            test_metric['rmse_f'],
            test_metric['rmse_cf']
        ))


def metrics(
        pred_0: np.ndarray,
        pred_1: np.ndarray,
        yf: np.ndarray,
        ycf: np.ndarray,
        t: np.ndarray,
        mode) -> dict:

    assert len(pred_0.shape) == 1
    assert len(pred_1.shape) == 1
    assert len(yf.shape) == 1 and len(ycf.shape) == 1
    assert len(t.shape) == 1
    from sklearn.metrics import r2_score, mean_squared_error

    length = len(t)
    y0 = yf * (1-t) + ycf * t
    y1 = yf * t + ycf * (1-t)

    # Section: factual fitting
    yf_pred = pred_1 * t + pred_0 * (1-t)
    r2_f = r2_score(yf, yf_pred)
    rmse_f = np.sqrt(mean_squared_error(yf, yf_pred))

    # Section: counterfactual fitting
    ycf_pred = pred_0 * t + pred_1 * (1-t)
    r2_cf = r2_score(ycf, ycf_pred)
    rmse_cf = np.sqrt(mean_squared_error(ycf, ycf_pred))

    # Section: ITE estimation
    _pred_0 = deepcopy(pred_0)
    _pred_1 = deepcopy(pred_1)
    if mode == "in-sample":
        _pred_0[t == 0] = y0[t == 0]
        _pred_1[t == 1] = y1[t == 1]
    effect_pred = _pred_1 - _pred_0
    effect = y1 - y0

    # Negative effect
    effect_pred = effect_pred
    effect = effect

    pehe = np.sqrt(np.mean((effect - effect_pred) ** 2))
    ate = np.mean(effect)
    ate_pred = np.mean(effect_pred)
    att = np.mean(effect[t == 1])
    att_pred = np.mean(effect_pred[t == 1])
    mae_ate = np.abs(ate - ate_pred)
    mae_att = np.abs(att - att_pred)
    auuc = auuc_score(yf=yf, t=t, effect_pred=effect_pred)


    return {
        "mae_ate": round(mae_ate, 5),
        "mae_att": round(mae_att, 5),
        "pehe": round(pehe, 5),
        "r2_f": round(r2_f, 5),
        "rmse_f": round(rmse_f, 5),
        "r2_cf": round(r2_cf, 5),
        "rmse_cf": round(rmse_cf, 5),
        "auuc": round(auuc[0], 5),
        "rauuc": round(auuc[1], 5)
    }

def metrics_tree(
        ite_pred: np.ndarray,
        yf: np.ndarray,
        ycf: np.ndarray,
        t: np.ndarray) -> dict:
    """
    Metric calculation for causal tree-based methods
    """
    assert len(yf.shape) == 1 and len(ycf.shape) == 1
    assert len(t.shape) == 1

    y0 = yf * (1-t) + ycf * t
    y1 = yf * t + ycf * (1-t)

    r2_f, rmse_f = 0, 0
    r2_cf, rmse_cf = 0, 0
    # Section: ITE estimation
    effect = y1 - y0
    # Negative effect
    effect_pred = ite_pred
    effect = effect
    pehe = np.sqrt(np.mean((effect - effect_pred) ** 2))
    ate = np.mean(effect)
    ate_pred = np.mean(effect_pred)
    att = np.mean(effect[t == 1])
    att_pred = np.mean(effect_pred[t == 1])
    mae_ate = np.abs(ate - ate_pred)
    mae_att = np.abs(att - att_pred)
    auuc = auuc_score(yf=yf, t=t, effect_pred=effect_pred)


    return {
        "mae_ate": round(mae_ate, 5),
        "mae_att": round(mae_att, 5),
        "pehe": round(pehe, 5),
        "r2_f": round(r2_f, 5),
        "rmse_f": round(rmse_f, 5),
        "r2_cf": round(r2_cf, 5),
        "rmse_cf": round(rmse_cf, 5),
        "auuc": round(auuc[0], 5),
        "rauuc": round(auuc[1], 5)
    }

import torch
import numpy as np
from simi_ite.propensity import *
import math
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from scipy import sparse
import argparse

# 模拟 TensorFlow flags 功能，实际使用通常结合 argparse
# class Flags:
#     def __init__(self):
#         self.sparse = 0  # 默认值，需根据实际情况调整
    
#     def flag_values_dict(self):
#         return self.__dict__

# FLAGS = Flags()
# 如果需要从命令行解析参数，可以使用 argparse 替换上面的 Flags 类

# SQRT_CONST = 1e-10

# def validation_split(D_exp, val_fraction):
#     """ Construct a train/validation split """
#     n = D_exp['x'].shape[0]

#     if val_fraction > 0:
#         n_valid = int(val_fraction*n)
#         n_train = n-n_valid
#         I = np.random.permutation(range(0,n))
#         I_train = I[:n_train]
#         I_valid = I[n_train:]
#     else:
#         I_train = range(n)
#         I_valid = []

#     return I_train, I_valid

# def validation_split_equal(D_exp, val_fraction):
#     """ Construct a train/validation split """
#     n = D_exp.shape[0]

#     if val_fraction > 0:
#         n_valid = int(val_fraction*n)
#         n_train = n-n_valid
#         I = np.random.permutation(range(0,n))
#         I_train = I[:n_train]
#         I_valid = I[n_train:]
#     else:
#         I_train = range(n)
#         I_valid = []

#     return I_train, I_valid

# def log(logfile, str_msg):
#     """ Log a string in a file """
#     with open(logfile, 'a') as f:
#         f.write(str_msg + '\n')
#     print(str_msg)

# def save_config(fname):
#     """ Save configuration """
#     # 假设 FLAGS 是一个 argparse.Namespace 或者上面的 Flags 类实例
#     try:
#         flagdict = FLAGS.flag_values_dict()
#     except AttributeError:
#         # 如果使用的是 argparse
#         flagdict = vars(FLAGS)
        
#     s = '\n'.join(['%s: %s' % (k, str(flagdict[k])) for k in sorted(flagdict.keys())])
#     with open(fname, 'w') as f:
#         f.write(s)

# def load_data(fname):
#     """ Load data set """
#     if fname[-3:] == 'npz':
#         data_in = np.load(fname)
#         data = {'x': data_in['x'], 't': data_in['t'], 'yf': data_in['yf']}
#         try:
#             data['ycf'] = data_in['ycf']
#         except:
#             data['ycf'] = None
#     else:
#         if FLAGS.sparse > 0:
#             data_in = np.loadtxt(open(fname+'.y', "rb"), delimiter=",")
#             x = load_sparse(fname+'.x')
#         else:
#             data_in = np.loadtxt(open(fname, "rb"), delimiter=",")
#             x = data_in[:, 5:]

#         data['x'] = x
#         data['t'] = data_in[:, 0:1]
#         data['yf'] = data_in[:, 1:2]
#         data['ycf'] = data_in[:, 2:3]

#     data['HAVE_TRUTH'] = not data['ycf'] is None

#     data['dim'] = data['x'].shape[1]
#     data['n'] = data['x'].shape[0]

#     return data

# def load_sparse(fname):
#     """ Load sparse data set """
#     E = np.loadtxt(open(fname, "rb"), delimiter=",")
#     H = E[0, :]
#     n = int(H[0])
#     d = int(H[1])
#     E = E[1:, :]
#     S = sparse.coo_matrix((E[:, 2], (E[:, 0]-1, E[:, 1]-1)), shape=(n, d))
#     S = S.todense() 

#     return S

# # 辅助函数
# def safe_sqrt(x, lbound=SQRT_CONST):
#     ''' Numerically safe version of sqrt  数值安全的开平方根函数。'''
#     return torch.sqrt(torch.clamp(x, min=lbound, max=float('inf')))

# def pdist2sq(X, Y):
#     """ Computes the squared Euclidean distance between all pairs x in X, y in Y  计算矩阵 X 和矩阵 Y 之间所有点对的平方欧几里得距离。"""
#     # C = -2 * X * Y.T
#     C = -2 * torch.matmul(X, Y.t())
#     nx = torch.sum(torch.square(X), dim=1, keepdim=True)
#     ny = torch.sum(torch.square(Y), dim=1, keepdim=True)
#     D = (C + ny.t()) + nx
#     return D

# def pdist2(X, Y):
#     """ Returns the pairwise distance matrix 计算矩阵 X 和矩阵 Y 之间所有点对的标准欧几里得距离。 """
#     return safe_sqrt(pdist2sq(X, Y))

# # 用于数据处理的函数
# def simplex_project(x, k):
#     """ Projects a vector x onto the k-simplex """
#     d = x.shape[0]
#     mu = np.sort(x, axis=0)[::-1]
#     nu = (np.cumsum(mu)-k)/range(1, d+1)
#     I = [i for i in range(0, d) if mu[i]>nu[i]]
#     theta = nu[I[-1]]
#     w = np.maximum(x-theta, 0)
#     return w

# def sigmoid(x):
#     return 1 / float(1 + math.exp(-x))

# 计算相似度得分
def similarity_score(s_i, s_j):
    # if mode == 'sigmoid':
    #     _mid = (s_i + s_j)/float(2)
    #     _dis = abs(s_j - s_i)/float(2)
    #     score = 2*sigmoid(abs(_mid-0.5)) - 3*sigmoid(_dis)+1
    # if mode == 'linear':
    _mid = (s_i + s_j) / float(2)
    _dis = abs(s_j - s_i) / float(2)
    score = (1.5 * abs(_mid - 0.5) - 2 * _dis + 1)/float(2)
    return score

# def propensity_dist(x, y):
#     s_x = load_propensity_score('./tmp/propensity_model.sav', x.reshape(1, x.shape[0]))
#     s_y = load_propensity_score('./tmp/propensity_model.sav', y.reshape(1, y.shape[0]))

#     edu_dist = np.power(np.linalg.norm(x-y), 2)
#     score = np.exp(-1*(1-similarity_score(s_x, s_y)) * edu_dist)
#     return score

# def square_dist(x, y):
#     dist = np.power(np.linalg.norm(x-y), 2)
#     return dist

# # 用于计算“相似度误差”
# def similarity_error_cal(x, h_rep_norm): #计算原始空间 $x$ 和 隐层空间 $h$ 之间的结构差异（KL 散度）。
#     distance_matrix_x = cdist(x, x, propensity_dist)
#     distance_matrix_h = cdist(h_rep_norm, h_rep_norm, "sqeuclidean")
#     dim = distance_matrix_h.shape[0]
#     il2 = np.tril_indices(dim, -1)
#     p_x = distance_matrix_x[il2]
#     p_x = p_x/sum(p_x)
#     p_h = distance_matrix_h[il2]
#     p_h = p_h / sum(p_h)
#     print(p_x)
#     print(p_h)
#     k_l = entropy(p_x, p_h)
#     return k_l

# 用于 PDDM/SITe 损失
def row_wise_dist(x):#高效计算矩阵 $x$ 内部样本两两之间的 平方欧几里得距离。
    r = torch.sum(x * x, dim=1)
    # turn r into column vector
    r = torch.reshape(r, [-1, 1])
    D = r - 2 * torch.matmul(x, x.t()) + r.t()
    return D

def get_simi_ground(x, propensity_dir='./simi_ite/tmp/propensity_model.sav'):#构建 “倾向性相似度”矩阵，作为训练的 Ground Truth（标准答案）。
    x_propensity_score = load_propensity_score(propensity_dir, x)
    n_train = x.shape[0]
    s_x_matrix = np.ones([n_train, n_train])
    for i in range(n_train):
        for j in range(n_train):
            s_x_matrix[i, j] = similarity_score(x_propensity_score[i], x_propensity_score[j])
    return s_x_matrix

def find_nearest_point(x, p):#在数组 $x$ 中，找到与数值 $p$ 最接近的另一个点的索引
    diff = np.abs(x-p)
    diff_1 = diff[diff>0]
    min_val = np.min(diff_1)
    I_diff = np.where(diff == min_val)[0]
    I_diff = I_diff[0]
    if I_diff.size > 1:
        I_diff = I_diff[0]
    return I_diff

def find_three_pairs(x, t, x_propensity_score):
    try:
        x_return = np.ones([6, x.shape[1]])
        I_x_return = np.zeros(6, dtype=int)
        # x_propensity_score = load_propensity_score(propensity_dir, x)
        I_t = np.where(t > 0)[0]
        I_c = np.where(t < 1)[0]

        prop_t = x_propensity_score[I_t]
        prop_c = x_propensity_score[I_c]
        
        x_t = x[I_t]
        x_c = x[I_c]
        
        # find x_i, x_j
        index_t, index_c = find_middle_pair(prop_t, prop_c)
        # find x_k, x_l
        index_k = np.argmax(np.abs(prop_c - prop_t[index_t]))
        index_l = find_nearest_point(prop_c, prop_c[index_k])

        # find x_n, x_m
        index_m = np.argmax(np.abs(prop_t - prop_c[index_c]))
        index_n = find_nearest_point(prop_t, prop_t[index_m,])
        
        x_return[0, :] = x_t[index_t, :]
        x_return[1, :] = x_c[index_c, :]
        x_return[2, :] = x_c[index_k, :]
        x_return[3, :] = x_c[index_l, :]
        x_return[4, :] = x_t[index_m, :]
        x_return[5, :] = x_t[index_n, :]
        I_x_return[0] = int(I_t[index_t])
        I_x_return[1] = int(I_c[index_c])
        I_x_return[2] = int(I_c[index_k])
        I_x_return[3] = int(I_c[index_l])
        I_x_return[4] = int(I_t[index_m])
        I_x_return[5] = int(I_t[index_n])
    except:
        x_return = x[0:6, :]
        I_x_return = np.array([0, 1, 2, 3, 4, 5])
        print('some error happens here!')

    return x_return, I_x_return

def find_middle_pair(x, y):
    min_val = np.abs(x[0]-0.5) + np.abs(y[0]-0.5)
    index_1 = 0
    index_2 = 0
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            value = np.abs(x[i]-0.5) + np.abs(y[j]-0.5)
            if value < min_val:
                min_val = value
                index_1 = i
                index_2 = j
    return index_1, index_2

# def get_three_pair_simi(three_pairs, file_dir='./simi_ite/tmp/propensity_model.sav'):
#     three_pairs_simi = get_simi_ground(three_pairs, file_dir)
#     simi = np.ones([5, 1])
#     '''
#     S(k, l), S(m, n), S(k, l), S(i, k), S(j, m)
#     '''
#     simi[0, 0] = three_pairs_simi[2, 3]
#     simi[1, 0] = three_pairs_simi[4, 5]
#     simi[2, 0] = three_pairs_simi[2, 4]
#     simi[3, 0] = three_pairs_simi[0, 2]
#     simi[4, 0] = three_pairs_simi[1, 4]
#     return simi
def get_three_pair_simi(similarity_ground, three_pairs_index):
    simi = np.ones([5, 1])
    '''
    S(k, l), S(m, n), S(k, l), S(i, k), S(j, m)
    '''
    simi[0, 0] = similarity_ground[three_pairs_index[2], three_pairs_index[3]]
    simi[1, 0] = similarity_ground[three_pairs_index[4], three_pairs_index[5]]
    simi[2, 0] = similarity_ground[three_pairs_index[2], three_pairs_index[4]]
    simi[3, 0] = similarity_ground[three_pairs_index[0], three_pairs_index[2]]
    simi[4, 0] = similarity_ground[three_pairs_index[1], three_pairs_index[4]]
    return simi

def metric_update(metric: dict(), metric_: dict(), epoch) -> dict():
    """
    Update the metric dict
    :param metric: self.metric in the class Estimator, each value is array
    :param metric_: output of metric() function, each value is float
    :return:
    """
    for key in metric_.keys():
        metric[key] = np.concatenate([metric[key], [metric_[key]]])
    info = "Epoch {:>3}".format(epoch)
    return metric


def metric_export(path, train_metric, eval_metric, test_metric):

    with open(path+'/run.txt', 'w') as f:
        f.write("r2_f,r2_cf\n")
        f.write("{},{}\n".format(
            'train',
            train_metric['r2_f'],
            train_metric['rmse_f'],
        ))
        f.write("{},{}\n".format(
            'eval',
            eval_metric['r2_f'],
            eval_metric['rmse_f']
        ))

        f.write("{},{}\n".format(
            'test',

            test_metric['r2_f'],
            test_metric['rmse_f']
        ))


def metrics(
        pred_0: np.ndarray,
        # pred_1: np.ndarray,
        yf: np.ndarray,
        # ycf: np.ndarray,
        # t: np.ndarray,
        ) -> dict:

    assert len(pred_0.shape) == 1
    # assert len(pred_1.shape) == 1
    assert len(yf.shape) == 1 and len(ycf.shape) == 1
    # assert len(t.shape) == 1
    from sklearn.metrics import r2_score, mean_squared_error

    # length = len(t)
    # y0 = yf * (1-t) + ycf * t
    # y1 = yf * t + ycf * (1-t)

    # Section: factual fitting
    # yf_pred = pred_1 * t + pred_0 * (1-t)
    r2_f = r2_score(yf,pred_0)
    rmse_f = np.sqrt(mean_squared_error(yf, pred_0))

    # Section: counterfactual fitting
    # ycf_pred = pred_0 * t + pred_1 * (1-t)
    # r2_cf = r2_score(ycf, ycf_pred)
    # rmse_cf = np.sqrt(mean_squared_error(ycf, ycf_pred))

    # Section: ITE estimation
    # _pred_0 = deepcopy(pred_0)
    # _pred_1 = deepcopy(pred_1)
    # if mode == "in-sample":
    #     _pred_0[t == 0] = y0[t == 0]
    #     _pred_1[t == 1] = y1[t == 1]
    # effect_pred = _pred_1 - _pred_0
    # effect = y1 - y0

    # # Negative effect
    # effect_pred = effect_pred
    # effect = effect

    # pehe = np.sqrt(np.mean((effect - effect_pred) ** 2))
    # ate = np.mean(effect)
    # ate_pred = np.mean(effect_pred)
    # att = np.mean(effect[t == 1])
    # att_pred = np.mean(effect_pred[t == 1])
    # mae_ate = np.abs(ate - ate_pred)
    # mae_att = np.abs(att - att_pred)
    # auuc = auuc_score(yf=yf, t=t, effect_pred=effect_pred)


    return {
        # "mae_ate": round(mae_ate, 5),
        # "mae_att": round(mae_att, 5),
        # "pehe": round(pehe, 5),
        "r2_f": round(r2_f, 5),
        "rmse_f": round(rmse_f, 5),
        # "r2_cf": round(r2_cf, 5),
        # "rmse_cf": round(rmse_cf, 5),
        # "auuc": round(auuc[0], 5),
        # "rauuc": round(auuc[1], 5)
    }

if __name__ == "__main__":
    pass