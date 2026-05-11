import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import spearmanr

def extract_upper_triangle(matrix):
    """提取矩阵的上三角部分并展平为一维向量，排除对角线"""
    # np.triu_indices_from 返回上三角的行、列索引，k=1 表示不包含对角线
    row_idx, col_idx = np.triu_indices_from(matrix, k=1)
    return matrix[row_idx, col_idx]

def mantel_test(rsm1, rsm2, permutations=100):
    """
    计算 Mantel Test (两个相似度矩阵上三角的 Spearman 秩相关系数)
    附带简单的置换检验 (Permutation Test) 计算 p 值
    """
    vec1 = extract_upper_triangle(rsm1)
    vec2 = extract_upper_triangle(rsm2)
    
    # 计算真实的 Spearman 相关系数
    r_true, _ = spearmanr(vec1, vec2)
    
    # 进行置换检验 (打乱其中一个矩阵的行列对应关系，看随机相关性是否能超过真实相关性)
    N = rsm1.shape[0]
    better_count = 0
    for _ in range(permutations):
        perm_idx = np.random.permutation(N)
        # 对 rsm2 的行和列进行相同的重排
        rsm2_perm = rsm2[perm_idx, :][:, perm_idx]
        vec2_perm = extract_upper_triangle(rsm2_perm)
        r_perm, _ = spearmanr(vec1, vec2_perm)
        
        if np.abs(r_perm) >= np.abs(r_true):
            better_count += 1
            
    p_value = (better_count + 1) / (permutations + 1)
    return r_true, p_value

def main():
    output_dir = "results/5"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data for RSA...")
    df = pd.read_csv("features/features_acoustic_norm.csv")
    
    # 1. 建立极其严格的过滤遮罩 (Mask)
    # 必须是合法元音，且 F1 和 F2 都不能是 NaN (否则声学 Euclidean 距离会报错)
    oral_vowels = ['a', 'i', 'u', 'y', 'e', 'ɛ', 'o', 'ɔ', 'ø', 'œ', 'ə']
    valid_mask = df['phoneme'].isin(oral_vowels) & df['F1_mid_norm'].notna() & df['F2_mid_norm'].notna()
    valid_indices = np.where(valid_mask)[0] # 获取原始行号
    
    # 2. 内存保护机制：随机下采样到 2500 个点
    N_SAMPLES = 2500
    if len(valid_indices) > N_SAMPLES:
        np.random.seed(42) # 设定随机种子，保证每次跑结果一致！
        sampled_indices = np.random.choice(valid_indices, size=N_SAMPLES, replace=False)
        sampled_indices = np.sort(sampled_indices) # 保持原有的时间顺序
    else:
        sampled_indices = valid_indices

    print(f"Constructing RSMs using {len(sampled_indices)} aligned tokens...")

    # 3. 提取特征并严格对齐
    # 声学: 使用 Lobanov 标准化后的 F1 和 F2
    acoustic_features = df.loc[sampled_indices, ['F1_mid_norm', 'F2_mid_norm']].values
    
    # 神经: 使用 50维 PCA 降维特征 (保留了丰富的表征空间，比 2 维更准确)
    whisper_npz = np.load("features/features_whisper_norm.npz")
    xlsr_npz = np.load("features/features_xlsr_norm.npz")
    
    # 选取模型特定的层 (这里以 Whisper 20层 和 XLS-R 18层 为例，你可以根据需求修改)
    whisper_features = whisper_npz['layer_20_pca50'][sampled_indices]
    xlsr_features = xlsr_npz['layer_18_pca50'][sampled_indices]

    # 4. 计算表征相似性矩阵 (RSMs)
    # 神经表征: Cosine Similarity (余弦相似度)
    rsm_whisper = cosine_similarity(whisper_features)
    rsm_xlsr = cosine_similarity(xlsr_features)
    
    # 声学表征: 负的欧几里得距离 (Negative Euclidean Distance)
    # 距离越小，相似度越大，所以加上负号与余弦相似度(越大越相似)保持方向一致
    dist_acoustic = euclidean_distances(acoustic_features)
    rsm_acoustic = -dist_acoustic

    # 5. 运行 Mantel Test
    print("Running Mantel tests (this may take a minute due to permutations)...")
    results = []
    
    comparisons = [
        ("Acoustic vs. Whisper (L20)", rsm_acoustic, rsm_whisper),
        ("Acoustic vs. XLS-R (L18)", rsm_acoustic, rsm_xlsr),
        ("Whisper (L20) vs. XLS-R (L18)", rsm_whisper, rsm_xlsr)
    ]
    
    for name, mat1, mat2 in comparisons:
        print(f"  Comparing: {name}")
        r, p = mantel_test(mat1, mat2, permutations=100)
        results.append({
            "Comparison": name,
            "Mantel r (Spearman)": round(r, 4),
            "p-value": "< 0.01" if p <= 0.01 else round(p, 4)
        })

    # 6. 保存结果
    rsm_df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, "rsa_mantel_results.csv")
    rsm_df.to_csv(output_path, index=False)
    
    display_df = rsm_df.to_string(index=False)
    print("\n--- Mantel Test Results ---")
    print(display_df)
    print(f"\nSuccess! Results saved to {output_path}")

if __name__ == "__main__":
    main()