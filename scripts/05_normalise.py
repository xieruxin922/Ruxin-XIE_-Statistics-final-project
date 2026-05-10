import os
import yaml
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap

def apply_lobanov(df):
    """
    Applies Lobanov normalisation (z-scoring per speaker).
    Crucial: The mean and standard deviation MUST be computed ONLY from vowel tokens to avoid bias from consonants.
    """
    df_norm = df.copy()
    
    # Define what constitutes a vowel in this corpus
    vowels = ['a', 'i', 'u', 'y', 'e', 'ɛ', 'o', 'ɔ', 'ø', 'œ', 'ə', 'ɑ̃', 'ɛ̃', 'ɔ̃', 'œ̃']
    
    for speaker_id, speaker_data in df.groupby('speaker_id'):
        # 1. 只提取该说话人的元音数据来计算“锚点”参考值
        speaker_vowels = speaker_data[speaker_data['phoneme'].isin(vowels)]
        idx = speaker_data.index
        
        # 2. 分别计算 F1, F2, F3 的全局参考值 (只看 Midpoint!)
        anchors = {
            'F1': (speaker_vowels['F1_mid'].mean(skipna=True), speaker_vowels['F1_mid'].std(skipna=True)),
            'F2': (speaker_vowels['F2_mid'].mean(skipna=True), speaker_vowels['F2_mid'].std(skipna=True)),
            'F3': (speaker_vowels['F3_mid'].mean(skipna=True), speaker_vowels['F3_mid'].std(skipna=True))
        }
        
        # 3. 开始执行标准化
        for formant in ['F1', 'F2', 'F3']:
            spk_mean, spk_std = anchors[formant]
            
            # 如果标准差无效（比如数据太少），跳过
            if pd.isna(spk_std) or spk_std == 0:
                continue
                
            # 把 Mid, 25, 75 都用同一个 Midpoint 的参考系进行标准化！
            for suffix in ['_mid', '_25', '_75']:
                col_name = f"{formant}{suffix}"
                norm_col_name = f"{col_name}_norm"
                
                # 检查这张表里到底有没有这一列 (兼容有和没有的长元音列)
                if col_name in df.columns:
                    # 空值 NaN 参与运算后依然是 NaN，完美规避短元音报错
                    df_norm.loc[idx, norm_col_name] = (df.loc[idx, col_name] - spk_mean) / spk_std
                    
    return df_norm


def reduce_dimensions(npz_path, output_path, dim_viz, dim_cluster):
    """
    Applies PCA (and UMAP for 2D) to the high-dimensional neural representations.
    """
    print(f"Processing {npz_path}...")
    data = np.load(npz_path)
    output_dict = {}
    
    for layer_name in data.files:
        matrix = data[layer_name]
        print(f"  Reducing {layer_name} with shape {matrix.shape}...")
        
        # 1. PCA for Clustering (e.g., d=50)
        # We use min() just in case the number of samples is smaller than the requested dimensions
        n_comp_cluster = min(dim_cluster, matrix.shape[0], matrix.shape[1])
        pca_cluster = PCA(n_components=n_comp_cluster)
        matrix_pca50 = pca_cluster.fit_transform(matrix)
        output_dict[f"{layer_name}_pca{dim_cluster}"] = matrix_pca50
        
        # 2. PCA for Visualisation (d=2)
        pca_viz = PCA(n_components=dim_viz)
        matrix_pca2 = pca_viz.fit_transform(matrix)
        output_dict[f"{layer_name}_pca{dim_viz}"] = matrix_pca2
        
        # 3. UMAP for Visualisation (d=2) - optional but recommended for non-linear structures
        print(f"    Running UMAP for {layer_name} ...")
        reducer = umap.UMAP(n_components=dim_viz, random_state=42)
        matrix_umap2 = reducer.fit_transform(matrix)
        output_dict[f"{layer_name}_umap{dim_viz}"] = matrix_umap2

    # Save to new NPZ file
    np.savez(output_path, **output_dict)
    print(f"  Saved reduced representations to {output_path}")

def main():
    # Load Parameters
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["normalisation"]
        
    dim_viz = params["pca_dim_viz"]
    dim_cluster = params["pca_dim_cluster"]

    # ---------------------------------------------------------
    # 1. Acoustic Features: Lobanov Normalisation
    # ---------------------------------------------------------
    print("Applying Lobanov normalisation to acoustic features...")
    df_acoustic = pd.read_csv("features/features_acoustic.csv")
    
    df_normed = apply_lobanov(df_acoustic)
    df_normed.to_csv("features/features_acoustic_norm.csv", index=False)
    print("Saved features_acoustic_norm.csv")

    # ---------------------------------------------------------
    # 2. Neural Features: Dimensionality Reduction
    # ---------------------------------------------------------
    # Whisper
    if os.path.exists("features/features_whisper.npz"):
        reduce_dimensions("features/features_whisper.npz", 
                          "features/features_whisper_norm.npz", 
                          dim_viz, dim_cluster)
                          
    # XLS-R
    if os.path.exists("features/features_xlsr.npz"):
        reduce_dimensions("features/features_xlsr.npz", 
                          "features/features_xlsr_norm.npz", 
                          dim_viz, dim_cluster)

    print("\nSuccess! Normalisation and Dimensionality Reduction complete.")

if __name__ == "__main__":
    main()