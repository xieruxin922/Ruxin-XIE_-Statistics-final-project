import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist, euclidean, cosine
from scipy.stats import spearmanr, chi2_contingency
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
import warnings
from sklearn.preprocessing import normalize

# 忽略因为个别 speaker 缺少某个音素导致的空切片警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

def extract_upper_triangle(matrix):
    row_idx, col_idx = np.triu_indices_from(matrix, k=1)
    return matrix[row_idx, col_idx]

def mantel_test(rsm1, rsm2):
    vec1 = extract_upper_triangle(rsm1)
    vec2 = extract_upper_triangle(rsm2)
    r_true, _ = spearmanr(vec1, vec2)
    return r_true

def compute_pooled_covariance(X, labels):
    """计算 Pooled within-phoneme covariance (用于马氏距离)"""
    n_features = X.shape[1]
    cov_pooled = np.zeros((n_features, n_features))
    total_dof = 0
    
    unique_labels = np.unique(labels)
    for c in unique_labels:
        X_c = X[labels == c]
        n_c = X_c.shape[0]
        if n_c > 1:
            cov_c = np.cov(X_c, rowvar=False)
            cov_pooled += (n_c - 1) * cov_c
            total_dof += (n_c - 1)
            
    cov_pooled /= total_dof
    return cov_pooled

def build_distance_matrix(features, labels, vowels, metric='euclidean', inv_cov=None):
    """计算音素质心之间的距离矩阵"""
    centroids = []
    valid_vowels = []
    
    for v in vowels:
        X_v = features[labels == v]
        if len(X_v) > 0:
            centroids.append(np.mean(X_v, axis=0))
            valid_vowels.append(v)
            
    centroids = np.array(centroids)
    
    if metric == 'mahalanobis' and inv_cov is not None:
        dist_mat = cdist(centroids, centroids, metric='mahalanobis', VI=inv_cov)
    else:
        dist_mat = cdist(centroids, centroids, metric=metric)
        
    return dist_mat, valid_vowels

def loso_cross_validation(features, labels, speaker_ids, metric='euclidean'):
    """留一说话人交叉验证 (Leave-One-Speaker-Out CV)"""
    speakers = np.unique(speaker_ids)
    y_true_all = []
    y_pred_all = []
    
    # 🚀 核心修复：如果是 cosine，先把特征做 L2 标准化，然后用 euclidean 偷梁换柱！
    if metric == 'cosine':
        features = normalize(features, norm='l2')
        actual_metric = 'euclidean'
    else:
        actual_metric = metric
        
    clf = NearestCentroid(metric=actual_metric)
    
    for test_spk in speakers:
        train_mask = speaker_ids != test_spk
        test_mask = speaker_ids == test_spk
        
        X_train, y_train = features[train_mask], labels[train_mask]
        X_test, y_test = features[test_mask], labels[test_mask]
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
            
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        
    return np.array(y_true_all), np.array(y_pred_all)


def mcnemar_test_models(y_true, y_pred1, y_pred2):
    """McNemar 检验比较两个模型的准确率 (匹配样本)"""
    correct1 = (y_true == y_pred1)
    correct2 = (y_true == y_pred2)
    
    n00 = np.sum(~correct1 & ~correct2)
    n01 = np.sum(~correct1 & correct2)
    n10 = np.sum(correct1 & ~correct2)
    n11 = np.sum(correct1 & correct2)
    
    table = [[n11, n10], [n01, n00]]
    result = mcnemar(table, exact=False, correction=True)
    return result.pvalue

def main():
    output_dir = "results/6"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading aligned dataset...")
    df = pd.read_csv("features/features_acoustic_norm.csv")
    oral_vowels = ['a', 'i', 'u', 'y', 'e', 'ɛ', 'o', 'ɔ', 'ø', 'œ', 'ə']
    
    # 严格对齐数据
    mask = df['phoneme'].isin(oral_vowels) & df['F1_mid_norm'].notna() & df['F2_mid_norm'].notna()
    df_valid = df[mask].copy()
    valid_idx = np.where(mask)[0]
    
    labels = df_valid['phoneme'].values
    speaker_ids = df_valid['speaker_id'].values
    l1_status = df_valid['l1_status'].values
    
    # 提取特征
    X_acoustic = df_valid[['F1_mid_norm', 'F2_mid_norm']].values
    
    npz_whisper = np.load("features/features_whisper_norm.npz")
    npz_xlsr = np.load("features/features_xlsr_norm.npz")
    X_whisper = npz_whisper['layer_20_pca50'][valid_idx]
    X_xlsr = npz_xlsr['layer_18_pca50'][valid_idx]

    # ==========================================
    # 1. Distance Matrices & Mantel Test
    # ==========================================
    print("Calculating Distance Matrices and Mantel Tests...")
    
    # 声学 Euclidean
    D_ac_euc, v_list = build_distance_matrix(X_acoustic, labels, oral_vowels, metric='euclidean')
    
    # 声学 Mahalanobis
    cov_pooled = compute_pooled_covariance(X_acoustic, labels)
    inv_cov = np.linalg.inv(cov_pooled)
    D_ac_mah, _ = build_distance_matrix(X_acoustic, labels, oral_vowels, metric='mahalanobis', inv_cov=inv_cov)
    
    # 神经 Cosine
    D_wh, _ = build_distance_matrix(X_whisper, labels, oral_vowels, metric='cosine')
    D_xl, _ = build_distance_matrix(X_xlsr, labels, oral_vowels, metric='cosine')
    
    results_mantel = [
        {"Comparison": "Acoustic (Euc) vs Whisper", "Mantel r": mantel_test(D_ac_euc, D_wh)},
        {"Comparison": "Acoustic (Euc) vs XLS-R", "Mantel r": mantel_test(D_ac_euc, D_xl)},
        {"Comparison": "Acoustic (Mah) vs Whisper", "Mantel r": mantel_test(D_ac_mah, D_wh)},
        {"Comparison": "Acoustic (Mah) vs XLS-R", "Mantel r": mantel_test(D_ac_mah, D_xl)},
        {"Comparison": "Whisper vs XLS-R", "Mantel r": mantel_test(D_wh, D_xl)}
    ]
    pd.DataFrame(results_mantel).to_csv(os.path.join(output_dir, "dist_mantel_results.csv"), index=False)

    # ==========================================
    # 2. Bootstrap CI for Selected Pairs
    # ==========================================
    print("Running Speaker-level Bootstrap CI (B=1000)...")
    pairs_to_test = [('e', 'ɛ'), ('y', 'u')]
    B = 1000
    boot_results = []
    
    unique_speakers = np.unique(speaker_ids)
    
    for p1, p2 in pairs_to_test:
        dists_ac, dists_wh, dists_xl = [], [], []
        
        for _ in range(B):
            # Speaker level resampling
            samp_spks = np.random.choice(unique_speakers, size=len(unique_speakers), replace=True)
            
            # 构建 bootstrap 切片
            boot_idx_list = []
            for spk in samp_spks:
                spk_idx = np.where(speaker_ids == spk)[0]
                boot_idx_list.extend(spk_idx)
            
            b_idx = np.array(boot_idx_list)
            b_labels = labels[b_idx]
            
            # 如果某次抽样恰好漏掉了 p1 或 p2，跳过
            if (p1 not in b_labels) or (p2 not in b_labels):
                continue
                
            c1_ac = np.mean(X_acoustic[b_idx][b_labels == p1], axis=0)
            c2_ac = np.mean(X_acoustic[b_idx][b_labels == p2], axis=0)
            dists_ac.append(euclidean(c1_ac, c2_ac))
            
            c1_wh = np.mean(X_whisper[b_idx][b_labels == p1], axis=0)
            c2_wh = np.mean(X_whisper[b_idx][b_labels == p2], axis=0)
            dists_wh.append(cosine(c1_wh, c2_wh))
            
            c1_xl = np.mean(X_xlsr[b_idx][b_labels == p1], axis=0)
            c2_xl = np.mean(X_xlsr[b_idx][b_labels == p2], axis=0)
            dists_xl.append(cosine(c1_xl, c2_xl))
            
        boot_results.append({
            "Pair": f"/{p1}/ - /{p2}/",
            "Acoustic Euc 95% CI": f"({np.percentile(dists_ac, 2.5):.3f}, {np.percentile(dists_ac, 97.5):.3f})",
            "Whisper Cos 95% CI": f"({np.percentile(dists_wh, 2.5):.3f}, {np.percentile(dists_wh, 97.5):.3f})",
            "XLS-R Cos 95% CI": f"({np.percentile(dists_xl, 2.5):.3f}, {np.percentile(dists_xl, 97.5):.3f})"
        })
        
    pd.DataFrame(boot_results).to_csv(os.path.join(output_dir, "dist_bootstrap_ci.csv"), index=False)

    # ==========================================
    # 3. Phoneme Identification (LOSO CV)
    # ==========================================
    print("Running Leave-One-Speaker-Out Nearest Centroid Classifiers...")
    
    y_true, y_pred_ac = loso_cross_validation(X_acoustic, labels, speaker_ids, metric='euclidean')
    _, y_pred_wh = loso_cross_validation(X_whisper, labels, speaker_ids, metric='cosine')
    _, y_pred_xl = loso_cross_validation(X_xlsr, labels, speaker_ids, metric='cosine')
    
    clf_results = [
        {"Model": "Acoustic", "Accuracy": accuracy_score(y_true, y_pred_ac), "Macro F1": f1_score(y_true, y_pred_ac, average='macro')},
        {"Model": "Whisper", "Accuracy": accuracy_score(y_true, y_pred_wh), "Macro F1": f1_score(y_true, y_pred_wh, average='macro')},
        {"Model": "XLS-R", "Accuracy": accuracy_score(y_true, y_pred_xl), "Macro F1": f1_score(y_true, y_pred_xl, average='macro')}
    ]
    pd.DataFrame(clf_results).to_csv(os.path.join(output_dir, "clf_performance.csv"), index=False)
    
    # McNemar Test 比较模型 (匹配样本)
    p_ac_wh = mcnemar_test_models(y_true, y_pred_ac, y_pred_wh)
    p_wh_xl = mcnemar_test_models(y_true, y_pred_wh, y_pred_xl)
    
    # 比较 L1 vs L2 准确率 (因为 L1和L2 是不同的人，不能用McNemar配对，必须用独立样本卡方检验)
    correct_wh = (y_true == y_pred_wh)
    l1_mask = (l1_status == 'fr')
    l2_mask = (l1_status == 'ru')
    contingency = [
        [np.sum(correct_wh[l1_mask]), np.sum(~correct_wh[l1_mask])],
        [np.sum(correct_wh[l2_mask]), np.sum(~correct_wh[l2_mask])]
    ]
    _, p_l1_l2, _, _ = chi2_contingency(contingency)
    
    stat_comparisons = [
        {"Comparison": "Acoustic vs Whisper Accuracy (McNemar)", "p-value": p_ac_wh},
        {"Comparison": "Whisper vs XLS-R Accuracy (McNemar)", "p-value": p_wh_xl},
        {"Comparison": "L1 vs L2 Accuracy in Whisper (Chi-Square)", "p-value": p_l1_l2}
    ]
    pd.DataFrame(stat_comparisons).to_csv(os.path.join(output_dir, "clf_statistical_tests.csv"), index=False)

    # 画 Confusion Matrix Heatmap (以 Whisper 为例)
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred_wh, labels=v_list)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=v_list, yticklabels=v_list)
    plt.title("Confusion Matrix (Whisper Nearest-Centroid LOSO)")
    plt.xlabel("Predicted Phoneme")
    plt.ylabel("True Phoneme")
    plt.savefig(os.path.join(output_dir, "fig_confusion_matrix_whisper.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("\nSuccess! Inter-phoneme distances and classification analyses are complete.")

if __name__ == "__main__":
    main()