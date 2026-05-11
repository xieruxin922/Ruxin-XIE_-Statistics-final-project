import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def calc_bcvr(X, labels):
    """计算 Between-class variance ratio (类间方差比)"""
    # 彻底剥离 Pandas 属性，变成纯 NumPy 数组
    labels = np.asarray(labels) 
    
    global_mean = np.mean(X, axis=0)
    sst = np.sum((X - global_mean)**2) # Total sum of squares
    
    ssb = 0 # Between-class sum of squares
    for c in np.unique(labels):
        Xc = X[labels == c]
        nc = len(Xc)
        if nc > 0:
            class_mean = np.mean(Xc, axis=0)
            ssb += nc * np.sum((class_mean - global_mean)**2)
        
    return ssb / sst if sst != 0 else np.nan

def calc_cos_sim_ratio(X, labels, sample_size=3000):
    """计算同类与异类的余弦相似度及其比值 (采用随机采样防止内存溢出)"""
    # 彻底剥离 Pandas 属性，变成纯 NumPy 数组
    labels = np.asarray(labels)
    
    n_samples = min(sample_size, len(X))
    idx = np.random.choice(len(X), n_samples, replace=False)
    
    X_samp = X[idx]
    labels_samp = labels[idx] # 这里不再需要 .iloc，因为已经是纯 Numpy 数组了
    
    # 计算两两之间的余弦相似度矩阵
    sim_matrix = cosine_similarity(X_samp)
    
    # 构建 Mask：利用 Numpy 原生广播机制 (彻底避开 Pandas 报错)
    same_mask = (labels_samp[:, None] == labels_samp[None, :])
    np.fill_diagonal(same_mask, False) 
    
    diff_mask = ~same_mask
    np.fill_diagonal(diff_mask, False)
    
    within_sim = np.mean(sim_matrix[same_mask])
    between_sim = np.mean(sim_matrix[diff_mask])
    
    ratio = within_sim / between_sim if between_sim != 0 else np.nan
    return within_sim, between_sim, ratio

def plot_2d_projections(X, df, title_prefix, output_path):
    """画出一排三个图: 分别按 Phoneme, L1_status, Gender 填色，并保存"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"{title_prefix} Projections", fontsize=16)
    
    # 1. By Phoneme
    palette_ph = sns.color_palette("husl", len(df['phoneme'].unique()))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=df['phoneme'], palette=palette_ph, s=15, alpha=0.7, ax=axes[0], edgecolor="none")
    axes[0].set_title("Colored by Phoneme")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
    
    # 2. By L1 Status
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=df['l1_status'], palette="Set1", s=15, alpha=0.7, ax=axes[1], edgecolor="none")
    axes[1].set_title("Colored by L1 Status")
    
    # 3. By Gender
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=df['gender'], palette="Set2", s=15, alpha=0.7, ax=axes[2], edgecolor="none")
    axes[2].set_title("Colored by Gender")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    output_dir = "results/5"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading metadata...")
    df_full = pd.read_csv("features/features_acoustic_norm.csv")
    
    # 定义合法的元音目标（与上一步保持一致，让图表更干净）
    oral_vowels = ['a', 'i', 'u', 'y', 'e', 'ɛ', 'o', 'ɔ', 'ø', 'œ', 'ə']
    mask = df_full['phoneme'].isin(oral_vowels).values
    df_filtered = df_full[mask].copy()

    results = []

    # 处理 Whisper 和 XLS-R
    models = {
        "Whisper": "features/features_whisper_norm.npz",
        "XLS-R": "features/features_xlsr_norm.npz"
    }

    for model_name, npz_path in models.items():
        if not os.path.exists(npz_path):
            print(f"Skipping {model_name}: {npz_path} not found.")
            continue
            
        print(f"Processing {model_name}...")
        data_npz = np.load(npz_path)
        
        # 遍历 NPZ 里存的所有矩阵 (提取我们做的 pca2 和 umap2)
        for key in data_npz.files:
            if not (key.endswith('_pca2') or key.endswith('_umap2')):
                continue # 跳过 50维的聚类矩阵
                
            layer_name = key.split('_')[1] # 提取层数，比如 '20'
            method = "UMAP" if "umap" in key else "PCA"
            
            # 必须用 mask 过滤矩阵，保证高维特征点和元音标签数量一致！
            X_filtered = data_npz[key][mask]
            
            # 1. 绘制并保存散点图
            plot_title = f"{model_name} Layer {layer_name} ({method})"
            plot_filename = os.path.join(output_dir, f"fig_{model_name.lower()}_l{layer_name}_{method.lower()}.png")
            print(f"  -> Plotting {plot_title}...")
            plot_2d_projections(X_filtered, df_filtered, plot_title, plot_filename)
            
            # 2. 计算 BCVR 和 余弦相似度
            print(f"  -> Calculating metrics for {plot_title}...")
            bcvr = calc_bcvr(X_filtered, df_filtered['phoneme'].values)
            w_sim, b_sim, sim_ratio = calc_cos_sim_ratio(X_filtered, df_filtered['phoneme'].values)
            
            results.append({
                "Model": model_name,
                "Layer": layer_name,
                "DR Method": method,
                "Variance Ratio (BCVR)": round(bcvr, 4),
                "Within-Phoneme Sim": round(w_sim, 4),
                "Between-Phoneme Sim": round(b_sim, 4),
                "Similarity Ratio": round(sim_ratio, 4)
            })

    # 将指标保存为 CSV
    metrics_df = pd.DataFrame(results)
    metrics_path = os.path.join(output_dir, "neural_metrics_comparison.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSuccess! Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()