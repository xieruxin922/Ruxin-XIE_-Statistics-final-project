import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize
import warnings

warnings.filterwarnings('ignore')

# 定义法语语音学先验知识 (Ground Truth)
phoneme_traits = {
    # 元音 (Vowels)
    'i': {'cv': 'V', 'fb': 'front', 'height': 'high'},
    'y': {'cv': 'V', 'fb': 'front', 'height': 'high'},
    'e': {'cv': 'V', 'fb': 'front', 'height': 'mid'},
    'ø': {'cv': 'V', 'fb': 'front', 'height': 'mid'},
    'ɛ': {'cv': 'V', 'fb': 'front', 'height': 'mid'},
    'œ': {'cv': 'V', 'fb': 'front', 'height': 'mid'},
    'a': {'cv': 'V', 'fb': 'front', 'height': 'low'},
    'u': {'cv': 'V', 'fb': 'back',  'height': 'high'},
    'o': {'cv': 'V', 'fb': 'back',  'height': 'mid'},
    'ɔ': {'cv': 'V', 'fb': 'back',  'height': 'mid'},
    'ə': {'cv': 'V', 'fb': 'front', 'height': 'mid'},
    # 辅音 (Consonants) - 挑选了不同发音部位和方法的代表
    'p': {'cv': 'C', 'fb': 'none', 'height': 'none'}, # 塞音
    't': {'cv': 'C', 'fb': 'none', 'height': 'none'},
    'k': {'cv': 'C', 'fb': 'none', 'height': 'none'},
    'f': {'cv': 'C', 'fb': 'none', 'height': 'none'}, # 擦音
    's': {'cv': 'C', 'fb': 'none', 'height': 'none'},
    'ʃ': {'cv': 'C', 'fb': 'none', 'height': 'none'},
    'm': {'cv': 'C', 'fb': 'none', 'height': 'none'}, # 响音: 鼻音
    'n': {'cv': 'C', 'fb': 'none', 'height': 'none'},
    'l': {'cv': 'C', 'fb': 'none', 'height': 'none'}, # 响音: 边音
    'ʁ': {'cv': 'C', 'fb': 'none', 'height': 'none'}, # 响音: 颤/擦音
    'j': {'cv': 'C', 'fb': 'none', 'height': 'none'}, # 响音: 滑音
    'w': {'cv': 'C', 'fb': 'none', 'height': 'none'}
}

def plot_dendrogram(Z, labels, title, filename):
    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=labels, leaf_rotation=0, leaf_font_size=12)
    plt.title(title)
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def extract_centroids(features, labels, vocab):
    """计算质心，带有极其严格的 NaN 防爆保护"""
    centroids = []
    valid_vocab = []
    for v in vocab:
        mask = (labels == v)
        if np.sum(mask) > 0:
            # 使用 nanmean 忽略 NaN 值计算均值
            c = np.nanmean(features[mask], axis=0)
            # 如果某个特征全都是 NaN (比如塞音没有F1)，用 0 填补，防止 linkage 报错
            c = np.nan_to_num(c) 
            centroids.append(c)
            valid_vocab.append(v)
    return np.array(centroids), np.array(valid_vocab)

def evaluate_clustering(features, labels_arr, vocab, feature_name, metric, output_dir, analysis_type="vowel"):
    """执行聚类并评估 ARI 和 Silhouette"""
    centroids, valid_vocab = extract_centroids(features, labels_arr, vocab)
    
    # 破解 Ward + Cosine 数学冲突陷阱
    if metric == 'cosine':
        centroids = normalize(centroids, norm='l2')
        Z = linkage(centroids, method='ward', metric='euclidean')
    else:
        # 声学特征直接使用 Euclidean
        Z = linkage(centroids, method='ward', metric='euclidean')
        
    plot_dendrogram(Z, valid_vocab, f"Dendrogram ({feature_name} - {analysis_type.upper()})", 
                    os.path.join(output_dir, f"fig_dendro_{analysis_type}_{feature_name}.png"))

    results = []
    
    if analysis_type == "vowel":
        gt_1 = [phoneme_traits[v]['fb'] for v in valid_vocab]
        gt_2 = [phoneme_traits[v]['height'] for v in valid_vocab]
        
        for k in range(2, 6):
            labels_pred = fcluster(Z, t=k, criterion='maxclust')
            sil = silhouette_score(centroids, labels_pred)
            results.append({
                "Representation": feature_name,
                "Analysis": "Oral Vowels",
                "k": k,
                "Silhouette": sil,
                "ARI (Front/Back)": adjusted_rand_score(gt_1, labels_pred),
                "ARI (Height)": adjusted_rand_score(gt_2, labels_pred)
            })
    
    elif analysis_type == "all_phonemes":
        gt_cv = [phoneme_traits[v]['cv'] for v in valid_vocab]
        labels_pred = fcluster(Z, t=2, criterion='maxclust') # C/V 边界通常看 k=2
        sil = silhouette_score(centroids, labels_pred)
        results.append({
            "Representation": feature_name,
            "Analysis": "Consonants vs Vowels",
            "k": 2,
            "Silhouette": sil,
            "ARI (C/V Boundary)": adjusted_rand_score(gt_cv, labels_pred)
        })
        
    return pd.DataFrame(results)

def cluster_speakers(df, features, vowels, feature_name, metric, output_dir):
    """9.3: 聚类 Speaker (带有严格的对齐与缺失值回退机制)"""
    speaker_ids = df['speaker_id'].unique()
    
    # 🚀 修复1：接收 valid_vocab，确保字典的键和值严格对应！
    global_centroids, valid_vocab = extract_centroids(features, df['phoneme'].values, vowels)
    global_dict = dict(zip(valid_vocab, global_centroids))
    
    # 获取特征的维度 (比如声学是2或4维，神经网络是50维)，用于生成全 0 替身
    feat_dim = features.shape[1] 
    
    profiles = []
    labels_l1 = []
    labels_gender = []
    
    for spk in speaker_ids:
        spk_idx = df.index[df['speaker_id'] == spk].tolist()
        if len(spk_idx) == 0: continue
        
        labels_l1.append(df.loc[spk_idx[0], 'l1_status'])
        labels_gender.append(df.loc[spk_idx[0], 'gender'])
        
        spk_profile = []
        for v in vowels:
            v_mask = (df['speaker_id'] == spk) & (df['phoneme'] == v)
            if v_mask.sum() > 0:
                c = np.nanmean(features[v_mask], axis=0)
                spk_profile.extend(np.nan_to_num(c))
            else:
                # 🚀 修复2：如果字典里连全局均值都找不到这个音，就用全 0 向量代替，绝对不报错！
                fallback = global_dict.get(v, np.zeros(feat_dim))
                spk_profile.extend(fallback)
                
        profiles.append(spk_profile)
        
    X_spk = np.array(profiles)
    
    if metric == 'cosine':
        X_spk = normalize(X_spk, norm='l2')
        Z = linkage(X_spk, method='ward', metric='euclidean')
    else:
        Z = linkage(X_spk, method='ward', metric='euclidean')
        
    labels_pred = fcluster(Z, t=2, criterion='maxclust')
    ari_l1 = adjusted_rand_score(labels_l1, labels_pred)
    ari_gender = adjusted_rand_score(labels_gender, labels_pred)
    
    plot_dendrogram(Z, labels_l1, f"Speaker Dendrogram ({feature_name}) - by L1", 
                    os.path.join(output_dir, f"fig_dendro_spk_{feature_name}.png"))
    
    return pd.DataFrame([{
        "Representation": feature_name,
        "Target k": 2,
        "ARI (L1 status)": ari_l1,
        "ARI (Gender)": ari_gender
    }])


def main():
    output_dir = "results/9"
    os.makedirs(output_dir, exist_ok=True)
    print("Loading data...")
    df = pd.read_csv("features/features_acoustic_norm.csv")
    
    # 获取所有的特征列 (包括可能的辅音特征)
    oral_vowels = ['a', 'i', 'u', 'y', 'e', 'ɛ', 'o', 'ɔ', 'ø', 'œ', 'ə']
    consonants = ['p', 't', 'k', 'f', 's', 'ʃ', 'm', 'n', 'l', 'ʁ', 'j', 'w']
    # 动态侦测你的 csv 里是否有这些辅音
    available_consonants = [c for c in consonants if c in df['phoneme'].unique()]
    all_phonemes = oral_vowels + available_consonants
    
    # 动态构建声学特征列 (F1, F2 加上可能存在的 Duration 和 COG)
    ac_cols = ['F1_mid_norm', 'F2_mid_norm', 'duration_ms', 'SCG']
    print(f"Using acoustic features: {ac_cols}")
    
    # 过滤出所有目标音素的数据
    mask = df['phoneme'].isin(all_phonemes)
    df_filtered = df[mask].copy().reset_index(drop=True)
    labels = df_filtered['phoneme'].values
    
    X_ac = df_filtered[ac_cols].values
    
    npz_whisper = np.load("features/features_whisper_norm.npz")
    npz_xlsr = np.load("features/features_xlsr_norm.npz")
    X_wh = npz_whisper['layer_20_pca50'][mask]
    X_xl = npz_xlsr['layer_18_pca50'][mask]
    
    # ==========================================
    # 9.1 Clustering of French Oral Vowels
    # ==========================================
    print("\n9.1 Running Vowel Clustering...")
    res_v_ac = evaluate_clustering(X_ac, labels, oral_vowels, "Acoustic", 'euclidean', output_dir, "vowel")
    res_v_wh = evaluate_clustering(X_wh, labels, oral_vowels, "Whisper", 'cosine', output_dir, "vowel")
    res_v_xl = evaluate_clustering(X_xl, labels, oral_vowels, "XLS-R", 'cosine', output_dir, "vowel")
    pd.concat([res_v_ac, res_v_wh, res_v_xl], ignore_index=True).to_csv(os.path.join(output_dir, "cluster_9_1_vowels.csv"), index=False)
    
    # ==========================================
    # 9.2 Consonants vs. Vowels
    # ==========================================
    if len(available_consonants) >= 6:
        print("\n9.2 Running Consonants vs. Vowels Clustering...")
        res_cv_ac = evaluate_clustering(X_ac, labels, all_phonemes, "Acoustic", 'euclidean', output_dir, "all_phonemes")
        res_cv_wh = evaluate_clustering(X_wh, labels, all_phonemes, "Whisper", 'cosine', output_dir, "all_phonemes")
        res_cv_xl = evaluate_clustering(X_xl, labels, all_phonemes, "XLS-R", 'cosine', output_dir, "all_phonemes")
        pd.concat([res_cv_ac, res_cv_wh, res_cv_xl], ignore_index=True).to_csv(os.path.join(output_dir, "cluster_9_2_cv_boundary.csv"), index=False)
    else:
        print("\nSkipping 9.2: Not enough consonants found in dataset.")

    # ==========================================
    # 9.3 Clustering of Speakers
    # ==========================================
    print("\n9.3 Running Speaker Clustering (Concatenating Vowels)...")
    # 作业要求仅拼接 vowels
    df_v = df_filtered[df_filtered['phoneme'].isin(oral_vowels)].reset_index(drop=True)
    mask_v = df_filtered['phoneme'].isin(oral_vowels)
    
    res_spk_ac = cluster_speakers(df_v, X_ac[mask_v], oral_vowels, "Acoustic", 'euclidean', output_dir)
    res_spk_wh = cluster_speakers(df_v, X_wh[mask_v], oral_vowels, "Whisper", 'cosine', output_dir)
    res_spk_xl = cluster_speakers(df_v, X_xl[mask_v], oral_vowels, "XLS-R", 'cosine', output_dir)
    pd.concat([res_spk_ac, res_spk_wh, res_spk_xl], ignore_index=True).to_csv(os.path.join(output_dir, "cluster_9_3_speakers.csv"), index=False)
    
    print("\nSuccess! All clustering analyses complete. Please check results/ folder.")

if __name__ == "__main__":
    main()