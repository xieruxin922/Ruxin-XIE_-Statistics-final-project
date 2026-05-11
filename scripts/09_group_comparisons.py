import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cosine
from statsmodels.stats.multitest import fdrcorrection

def test_acoustic_l1_l2(df, vowels):
    """
    1. L1 vs L2 on Acoustic Features (F1_mid_norm)
    包含正态性检验、方差齐性检验和自适应检验选择
    """
    results = []
    
    # 准备画 Q-Q Plot 的画布 (只画前4个元音作为代表展示在报告中)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Q-Q Plots for F1_norm (L1 vs L2)", fontsize=14)
    axes = axes.flatten()
    plot_idx = 0
    
    for phoneme in vowels:
        df_p = df[(df['phoneme'] == phoneme) & (df['F1_mid_norm'].notna())]
        l1_data = df_p[df_p['l1_status'] == 'fr']['F1_mid_norm'].values
        l2_data = df_p[df_p['l1_status'] == 'ru']['F1_mid_norm'].values
        
        if len(l1_data) < 5 or len(l2_data) < 5:
            continue
            
        # 1. Check Assumptions (假设检验)
        # Shapiro-Wilk (正态性)
        _, p_shapiro_l1 = stats.shapiro(l1_data)
        _, p_shapiro_l2 = stats.shapiro(l2_data)
        # Levene's (方差齐性)
        _, p_levene = stats.levene(l1_data, l2_data)
        
        assumptions_met = (p_shapiro_l1 > 0.05) and (p_shapiro_l2 > 0.05) and (p_levene > 0.05)
        
        # 2. Select and run test
        if assumptions_met:
            test_used = "t-test"
            stat, p_val = stats.ttest_ind(l1_data, l2_data, equal_var=True)
        else:
            test_used = "Mann-Whitney U"
            stat, p_val = stats.mannwhitneyu(l1_data, l2_data, alternative='two-sided')
            
        results.append({
            "Phoneme": phoneme,
            "N (L1/L2)": f"{len(l1_data)}/{len(l2_data)}",
            "Shapiro p (min)": min(p_shapiro_l1, p_shapiro_l2),
            "Levene p": p_levene,
            "Test Used": test_used,
            "Statistic": stat,
            "p-value (raw)": p_val
        })
        
        # 挑选几个音素画 Q-Q Plot
        if plot_idx < 4:
            stats.probplot(l1_data, dist="norm", plot=axes[plot_idx])
            axes[plot_idx].set_title(f"Phoneme /{phoneme}/ (L1)")
            plot_idx += 1
            
    plt.tight_layout()
    plt.savefig("results/6/fig_qq_plots.png", dpi=300)
    plt.close()
    
    res_df = pd.DataFrame(results)
    # 3. Apply Benjamini-Hochberg FDR correction
    # 把校正后的 p 值加到表格里
    rejected, pvals_corrected = fdrcorrection(res_df['p-value (raw)'], alpha=0.05, method='indep')
    res_df['p-value (FDR)'] = pvals_corrected
    res_df['Significant?'] = rejected
    
    return res_df

def test_gender_residual(df, vowels):
    """
    2. Gender Differences (Paired test)
    在 Lobanov 标准化后，检验男性和女性的残差是否存在显著差异。
    因为无法直接将独立的男女匹配，这里我们在“音素层面”进行配对：
    比较男性对各个音素的均值向量 vs 女性的均值向量 (Paired t-test)
    """
    df_v = df[df['phoneme'].isin(vowels)].copy()
    
    # 算出每个音素，男性的平均 F1 和女性的平均 F1
    gender_means = df_v.groupby(['phoneme', 'gender'])['F1_mid_norm'].mean().unstack()
    gender_means = gender_means.dropna()
    
    # 配对 t 检验
    stat, p_val = stats.ttest_rel(gender_means['m'], gender_means['f'])
    
    return pd.DataFrame([{
        "Test": "Paired t-test on normalized F1 across phonemes",
        "Statistic": stat,
        "p-value": p_val,
        "Significant (alpha=0.05)?": p_val < 0.05
    }])

def permutation_test_neural(df, neural_features, vowels, B=5000):
    """
    3. L1 vs L2 on Neural Representations (Permutation test on cosine distance)
    """
    results = []
    
    for phoneme in vowels:
        idx_p = df.index[df['phoneme'] == phoneme].tolist()
        if len(idx_p) < 10:
            continue
            
        df_p = df.loc[idx_p]
        features_p = neural_features[idx_p]
        labels_p = df_p['l1_status'].values
        
        l1_idx = np.where(labels_p == 'fr')[0]
        l2_idx = np.where(labels_p == 'ru')[0]
        
        if len(l1_idx) == 0 or len(l2_idx) == 0:
            continue
            
        # 1. 真实情况的 Cosine Distance (1 - cosine_similarity)
        # 求特征矩阵的列平均，得到质心 (Centroid)
        centroid_l1 = np.mean(features_p[l1_idx], axis=0)
        centroid_l2 = np.mean(features_p[l2_idx], axis=0)
        true_dist = cosine(centroid_l1, centroid_l2)
        
        # 2. 蒙特卡洛置换检验 (Permutation)
        n_l1 = len(l1_idx)
        n_total = len(labels_p)
        better_count = 0
        
        # 生成一个随机状态生成器加快速度
        rng = np.random.default_rng(42)
        
        for _ in range(B):
            # 打乱索引
            shuffled_indices = rng.permutation(n_total)
            pseudo_l1_idx = shuffled_indices[:n_l1]
            pseudo_l2_idx = shuffled_indices[n_l1:]
            
            pseudo_c_l1 = np.mean(features_p[pseudo_l1_idx], axis=0)
            pseudo_c_l2 = np.mean(features_p[pseudo_l2_idx], axis=0)
            pseudo_dist = cosine(pseudo_c_l1, pseudo_c_l2)
            
            # 检验原假设：真正的距离是不是比随机打乱得出的距离更极端
            if pseudo_dist >= true_dist:
                better_count += 1
                
        # 计算 p 值 (加 1 平滑法)
        p_val = (better_count + 1) / (B + 1)
        
        results.append({
            "Phoneme": phoneme,
            "True Cosine Dist": true_dist,
            "Permutation p-value": p_val
        })
        
    res_df = pd.DataFrame(results)
    # 3. Apply BH correction
    rejected, pvals_corrected = fdrcorrection(res_df['Permutation p-value'], alpha=0.05, method='indep')
    res_df['p-value (FDR)'] = pvals_corrected
    res_df['Significant?'] = rejected
    
    return res_df

def main():
    os.makedirs("results", exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv("features/features_acoustic_norm.csv")
    oral_vowels = ['a', 'i', 'u', 'y', 'e', 'ɛ', 'o', 'ɔ', 'ø', 'œ', 'ə']
    
    # 1. Acoustic L1 vs L2
    print("1. Running Acoustic L1 vs L2 tests (with assumption checking and FDR)...")
    res_acoustics = test_acoustic_l1_l2(df, oral_vowels)
    res_acoustics.to_csv("results/6/stat_acoustic_l1_l2.csv", index=False)
    
    # 2. Gender residual test
    print("2. Running Gender residual paired test...")
    res_gender = test_gender_residual(df, oral_vowels)
    res_gender.to_csv("results/6/stat_gender_residual.csv", index=False)
    
    # 3. Neural L1 vs L2 Permutation Test
    print("3. Running Neural Permutation Tests (B=5000, this may take a couple of minutes)...")
    # 我们以 Whisper 20 层的 50维 PCA 为例进行假设检验
    npz_data = np.load("features/features_whisper_norm.npz")
    neural_features = npz_data['layer_20_pca50'] 
    
    res_neural = permutation_test_neural(df, neural_features, oral_vowels, B=5000)
    res_neural.to_csv("results/6/stat_neural_l1_l2.csv", index=False)
    
    print("\nSuccess! All statistical tests completed and saved to results/.")

if __name__ == "__main__":
    main()