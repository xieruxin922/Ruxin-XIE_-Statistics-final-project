import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)

def cv(x):
    mean_val = x.mean()
    return x.std() / mean_val if mean_val != 0 else np.nan

def main():
    # 1. 准备输出文件夹
    output_dir = "results/5"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ensured at: {output_dir}/")

    # 2. 加载数据并预处理
    print("Loading normalised acoustic features...")
    df = pd.read_csv("features/features_acoustic_norm.csv")
    df['speaker_group'] = df['l1_status'] + "_" + df['gender']
    
    oral_vowels = ['a', 'i', 'u', 'y', 'e', 'ɛ', 'o', 'ɔ', 'ø', 'œ', 'ə', 'ɑ̃', 'ɛ̃', 'ɔ̃', 'œ̃']
    df_vowels = df[df['phoneme'].isin(oral_vowels)].copy()

    # ==========================================
    # 模块 A: 计算描述性统计 (Summary Stats)
    # ==========================================
    print("Calculating summary statistics...")
    stats_norm = df_vowels.groupby(['phoneme', 'speaker_group'])[['F1_mid_norm', 'F2_mid_norm']].agg(['mean', 'median', 'std', iqr])
    stats_raw = df_vowels.groupby(['phoneme', 'speaker_group'])[['F1_mid', 'F2_mid']].agg([cv])
    summary_table = pd.concat([stats_norm, stats_raw], axis=1)
    
    table_path = os.path.join(output_dir, "summary_statistics.csv")
    summary_table.to_csv(table_path)
    print(f" -> Saved {table_path}")

    # ==========================================
    # 模块 B: 方差分解 (Variance Decomposition LMM)
    # ==========================================
    print("Running variance decomposition...")
    variance_results = []
    for phoneme in oral_vowels:
        df_p = df_vowels[df_vowels['phoneme'] == phoneme].dropna(subset=['F1_mid_norm'])
        if len(df_p) < 10: 
            continue
        try:
            md = smf.mixedlm("F1_mid_norm ~ 1", df_p, groups=df_p["speaker_id"], re_formula="~1")
            mdf = md.fit(method='lbfgs')
            variance_results.append({
                'Phoneme': phoneme,
                'Total Var': df_p['F1_mid_norm'].var(),
                'Inter-Speaker Var': mdf.cov_re.iloc[0, 0],
                'Residual/Intra Var': mdf.scale
            })
        except Exception:
            pass # 静默跳过无法拟合的音素
            
    var_df = pd.DataFrame(variance_results)
    var_path = os.path.join(output_dir, "variance_decomposition.csv")
    var_df.to_csv(var_path, index=False)
    print(f" -> Saved {var_path}")

    # ==========================================
    # 模块 C: 生成并保存图表 (Visualisations)
    # ==========================================
    print("Generating visualisations...")
    sns.set_theme(style="whitegrid")

    # 1. 经典元音图
    plt.figure(figsize=(12, 9))
    sns.kdeplot(data=df_vowels, x='F2_mid_norm', y='F1_mid_norm', hue='phoneme', alpha=0.4, thresh=0.05, levels=2, linewidths=1.5, legend=False)
    
    speaker_groups = df_vowels['speaker_group'].unique()
    palette = sns.color_palette("Set1", n_colors=len(speaker_groups))
    for idx, group in enumerate(speaker_groups):
        subset = df_vowels[df_vowels['speaker_group'] == group]
        centroids = subset.groupby('phoneme')[['F2_mid_norm', 'F1_mid_norm']].mean().reset_index()
        sns.scatterplot(data=centroids, x='F2_mid_norm', y='F1_mid_norm', s=150, color=palette[idx], marker='X', label=f"{group} Centroid", zorder=5)
        for _, row in centroids.iterrows():
            plt.text(row['F2_mid_norm'] + 0.03, row['F1_mid_norm'] + 0.03, row['phoneme'], color=palette[idx], fontsize=13, fontweight='bold', zorder=6)
            
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.title("Vowel Space: Overall Phoneme Distributions vs. Group Centroids", fontsize=15)
    plt.xlabel("F2 (Front-Back / Normalized)", fontsize=12)
    plt.ylabel("F1 (Close-Open / Normalized)", fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    
    vowel_plot_path = os.path.join(output_dir, "fig_vowel_space.png")
    plt.savefig(vowel_plot_path, dpi=300, bbox_inches='tight')
    plt.close() # 必须 close，释放内存
    print(f" -> Saved {vowel_plot_path}")

    # 2. 箱线图
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_vowels, x='phoneme', y='F1_mid_norm', hue='speaker_group', palette="Set2")
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_vowels, x='phoneme', y='F2_mid_norm', hue='speaker_group', palette="Set2")
    plt.tight_layout()
    
    box_plot_path = os.path.join(output_dir, "fig_boxplots.png")
    plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved {box_plot_path}")

    # 3. 个体内变异图
    subset_intra = df_vowels[(df_vowels['phoneme'] == 'a') & (df_vowels['speaker_id'].isin(['AB', 'AB2','AG']))]
    if not subset_intra.empty:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=subset_intra, x='speaker_id', y='F1_mid_norm', inner=None, color=".8")
        sns.stripplot(data=subset_intra, x='speaker_id', y='F1_mid_norm', hue='repetition', size=8, jitter=True, palette="tab10")
        plt.title("Intra-speaker Variability for phoneme /a/ across 6 repetitions")
        
        intra_plot_path = os.path.join(output_dir, "fig_intra_speaker.png")
        plt.savefig(intra_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" -> Saved {intra_plot_path}")

    print("\nSuccess! All analyses completed and saved to results/.")

if __name__ == "__main__":
    main()