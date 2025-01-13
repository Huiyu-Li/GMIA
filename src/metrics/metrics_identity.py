import numpy as np
import sklearn
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]

def calculate_f1_score(threshold, dist, actual_issame, all=False):
    # Return the truth value of (x1 < x2) element-wise.
    predict_issame = np.less(dist, threshold) # (5400,)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    f1_score = (float(2*tp)) / float(2*tp + fp + fn)
    
    # print(f'tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn}, total: {dist.size}')
    if all:
        precision = 0 if (tp ==0) else float(tp) / float(tp + fp)
        recall  = 0 if (tp ==0) else float(tp) / float(tp + fn)
        precision = float(tp) / float(tp + fp)
        recall  = float(tp) / float(tp + fn)
        acc = float(tp + tn) / dist.size
        return f1_score, precision, recall, acc
    else:
        return f1_score
    
def get_threshold(thresholds, dist, actual_issame):
    # Find the best threshold for the fold
    nrof_thresholds = len(thresholds) # 400
    f1_score_valid = np.zeros((nrof_thresholds)) # 400
    for threshold_idx, threshold in enumerate(thresholds):
        f1_score_valid[threshold_idx] = calculate_f1_score(threshold, dist, actual_issame)
    
    best_threshold_index = np.argmax(f1_score_valid)
    best_threshold = thresholds[best_threshold_index]
    best_f1_score = f1_score_valid[best_threshold_index]
    # print(f'best threshold:{best_threshold}, best f1_score:{best_f1_score:.3f}')
    _, precision, recall, acc = calculate_f1_score(best_threshold, dist, actual_issame, all=True)
    # print(f'best 1_score:{f1_score:.3f}, precision:{precision:.3f}, recall:{recall:.3f}, acc:{acc:.3f}')

    return best_threshold, best_f1_score, precision, recall, acc

def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0]) # 6000
    nrof_thresholds = len(thresholds) # 400
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds)) # (10, 400)
    fprs = np.zeros((nrof_folds, nrof_thresholds)) # (10, 400)
    accuracy = np.zeros((nrof_folds)) # (10）
    indices = np.arange(nrof_pairs) # 6000

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2) # (6000, 512)
        dist = np.sum(np.square(diff), 1) # # (6000)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds)) # 400
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set], actual_issame[test_set])
        
        _, _, accuracy[fold_idx] = calculate_accuracy(
                thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
        # print(f'best_threshold={thresholds[best_threshold_index]}')
    tpr = np.mean(tprs, 0) # (10, 400)->(400)
    fpr = np.mean(fprs, 0) # (10, 400)->(400)
    return tpr, fpr, accuracy # (10）


def calculate_accuracy(threshold, dist, actual_issame):
    # Return the truth value of (x1 < x2) element-wise.
    predict_issame = np.less(dist, threshold) # (5400,)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    # print(f'tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}')
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    # print(f'tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn}, total: {dist.size}')
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target=1e-3,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0]) # 6000
    nrof_thresholds = len(thresholds) # 4000
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds) # 10
    far = np.zeros(nrof_folds) # 10
    frr = np.zeros(nrof_folds) # 10
    ACA = np.zeros(nrof_folds) # 10

    diff = np.subtract(embeddings1, embeddings2) #（6000, 512）
    dist = np.sum(np.square(diff), 1) #（6000,)
    indices = np.arange(nrof_pairs)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # print(f'train_set: {train_set.shape}, test_set: {test_set.shape}')
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds) #（4000,)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx], _, _ = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
        # print(f'=========> threshold: {threshold}')
        val[fold_idx], far[fold_idx], frr[fold_idx], ACA[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val); val_std = np.std(val)# 10->1
    far_mean = np.mean(far); far_std = np.std(far)# 10->1
    frr_mean = np.mean(frr); frr_std = np.std(frr)# 10->1
    ACA_mean = np.mean(ACA); ACA_std = np.std(ACA)# 10->1
    return val_mean, val_std, far_mean, far_std, frr_mean, frr_std, ACA_mean, ACA_std

def calculate_val_far(threshold, dist, actual_issame):
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))

    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    # print(f'{true_accept}/{n_same}, {false_accept}/{n_diff}')

    # validation rate/true accept rate (val/tar)
    tar = float(tp) / float(n_same)
    # False Accept Rate (FAR) is the probability that a natural fingerprint is wrongly identified as a fake fingerprint
    far = float(fn) / float(n_same)
    # False Reject Rate(FRR) is the probability of a fake fingerprint being misplaced as a natural fingerprint.
    # print(f'fp:{fp}, n_diff:{n_diff}')
    # frr = 1 if n_diff==0 else float(fp) / float(n_diff)
    frr =  float(fp) / float(n_diff)
    # Average Classification Accuracy (ACA), Average Classification Accuracy (ACE) 
    ACE = (far+frr)/2
    ACA = 1-ACE
    return tar, far, frr, ACA