import numpy as np
from config import * #noqa

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    mean = np.mean(y_true)
    ss_total = np.sum((y_true - mean)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    return 1 - ss_res / ss_total

def sq_dist(a,b):
    d = 0
    for i in range(len(a)):
        d = d + (a[i]-b[i])**2    
    return d

def precision_at_k(y_true, y_pred, k=TOP_K, threshold=RELEVANCE_THRESHOLD):  #noqa

    idx = np.argsort(-y_pred.reshape(-1))[:k]

    relevant = (y_true.reshape(-1) >= threshold)
    recommended = np.zeros_like(relevant)
    recommended[idx] = True

    tp = np.sum(relevant & recommended)

    return tp / k

def recall_at_k(y_true, y_pred, k=TOP_K, threshold=RELEVANCE_THRESHOLD):   #noqa

    idx = np.argsort(-y_pred.reshape(-1))[:k]

    relevant = (y_true.reshape(-1) >= threshold)
    recommended = np.zeros_like(relevant)
    recommended[idx] = True

    tp = np.sum(relevant & recommended)
    total_relevant = np.sum(relevant)

    if total_relevant == 0:
        return 0.0

    return tp / total_relevant

def hit_rate_at_k(y_true, y_pred, k=TOP_K, threshold=RELEVANCE_THRESHOLD):   #noqa

    idx = np.argsort(-y_pred.reshape(-1))[:k]
    relevant = (y_true.reshape(-1) >= threshold)

    return int(np.any(relevant[idx]))

def ndcg_at_k(y_true, y_pred, k=TOP_K):  #noqa

    idx = np.argsort(-y_pred.reshape(-1))[:k]
    gains = y_true.reshape(-1)[idx]

    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains / discounts)

    ideal_idx = np.argsort(-y_true.reshape(-1))[:k]
    ideal_gains = y_true.reshape(-1)[ideal_idx]
    idcg = np.sum(ideal_gains / discounts)

    if idcg == 0:
        return 0.0

    return dcg / idcg