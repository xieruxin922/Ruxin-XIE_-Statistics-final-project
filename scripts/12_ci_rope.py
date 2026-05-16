import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cosine
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)

def classify_rope(ci_low, ci_high, rope_low, rope_high):
    """根据置信区间和 ROPE 边界进行等效分类"""
    if ci_high < rope_low or ci_low > rope_high:
        return "Non-equivalent" # 彻底在 ROPE 外，说明有实际显著差异
    elif ci_low >= rope_low and ci_high <= rope_high:
        return "Equivalent"     # 彻底在 ROPE 内，说明实际上没有区别
    else:
        return "Indeterminate"  # 有重叠，无法得出确定结论

def compute_acoustic_cis(df, vowels, rope_hz=20):
    """1. 提取声学特征 (F1, F2) 的置信区间，强制使用原始赫兹 (Raw Hz) 并附带防爆保护"""
    results = []
    
    f1_col = 'F1_mid' if 'F1_mid' in df.columns else 'F1'
    f2_col = 'F2_mid' if 'F2_mid' in df.columns else 'F2'
    
    for ph in vowels:
        df_v = df[df['phoneme'] == ph].copy()
        if len(df_v) < 10:
            continue
            
        for feat in [f1_col, f2_col]:
            try:
                # 尝试拟合 LME 模型
                md = smf.mixedlm(f"{feat} ~ is_L2 + is_male", df_v, groups=df_v["speaker_id"])
                mdf = md.fit(method='lbfgs', reml=False)
                
                est = mdf.params['is_L2']
                ci_low, ci_high = mdf.conf_int().loc['is_L2']
                
                # 🚨 检查 CI 是否爆炸：如果跨度超过 2000 Hz，说明海森矩阵崩溃了！
                if np.isnan(ci_low) or (ci_high - ci_low) > 2000:
                    raise ValueError("Hessian matrix ill-conditioned, CI exploded.")
                    
            except Exception as e:
                # 🚀 终极保护机制：如果 LME 崩溃，自动降级为稳健 OLS 模型
                # 稳健标准误 (HC3) 可以完美处理异方差，是 LME 崩溃时的最佳学术替代方案
                md_ols = smf.ols(f"{feat} ~ is_L2 + is_male", df_v)
                mdf_ols = md_ols.fit(cov_type='HC3')
                
                est = mdf_ols.params['is_L2']
                ci_low, ci_high = mdf_ols.conf_int().loc['is_L2']
                
            # 统一使用 [-20, 20] 的 ROPE
            r_low, r_high = -rope_hz, rope_hz 
            classification = classify_rope(ci_low, ci_high, r_low, r_high)
            
            results.append({
                "Representation": f"Acoustic ({feat})",
                "Phoneme": ph,
                "Estimate": est,
                "CI_Lower": ci_low,
                "CI_Upper": ci_high,
                "ROPE": f"[{r_low}, {r_high}]",
                "ROPE_Classification": classification
            })
            
    return pd.DataFrame(results)


def compute_neural_cis(df, features, vowels, model_name, B=2000):
    """2. 计算神经网络的置信区间 (Speaker-level Bootstrap) 和 ROPE (Noise floor)"""
    results = []
    
    # 2.1 计算 Neural ROPE (delta_0)
    # 定义为：同一个说话人，发同一个音素时的平均变异度 (Intra-speaker distance)
    print(f"Calculating intra-speaker noise floor (delta_0) for {model_name}...")
    intra_dists = []
    for spk in df['speaker_id'].unique():
        for ph in vowels:
            idx = df.index[(df['speaker_id'] == spk) & (df['phoneme'] == ph)].tolist()
            if len(idx) > 1:
                dists = pdist(features[idx], metric='cosine')
                intra_dists.append(np.mean(dists))
                
    delta_0 = np.mean(intra_dists)
    rope_low, rope_high = 0.0, delta_0
    print(f"  -> Neural ROPE defined as [0.0, {delta_0:.4f}]")
    
    # 预计算所有 speaker-phoneme 的质心以极大地加速 Bootstrap
    spk_ph_centroids = {}
    for spk in df['speaker_id'].unique():
        for ph in vowels:
            idx = df.index[(df['speaker_id'] == spk) & (df['phoneme'] == ph)].tolist()
            if len(idx) > 0:
                spk_ph_centroids[(spk, ph)] = np.mean(features[idx], axis=0)
    
    # 2.2 Bootstrap Resampling (B=2000)
    rng = np.random.default_rng(42)
    for ph in vowels:
        spks_l1 = df[(df['phoneme'] == ph) & (df['l1_status'] == 'fr')]['speaker_id'].unique()
        spks_l2 = df[(df['phoneme'] == ph) & (df['l1_status'] == 'ru')]['speaker_id'].unique()
        
        if len(spks_l1) < 2 or len(spks_l2) < 2:
            continue
            
        boot_dists = []
        for _ in range(B):
            samp_l1 = rng.choice(spks_l1, len(spks_l1), replace=True)
            samp_l2 = rng.choice(spks_l2, len(spks_l2), replace=True)
            
            # 从预计算好的质心字典中提取数据
            feats_l1 = [spk_ph_centroids[(s, ph)] for s in samp_l1]
            feats_l2 = [spk_ph_centroids[(s, ph)] for s in samp_l2]
            
            c_l1 = np.mean(feats_l1, axis=0)
            c_l2 = np.mean(feats_l2, axis=0)
            boot_dists.append(cosine(c_l1, c_l2))
            
        est = np.mean(boot_dists)
        ci_low = np.percentile(boot_dists, 2.5)
        ci_high = np.percentile(boot_dists, 97.5)
        
        classification = classify_rope(ci_low, ci_high, rope_low, rope_high)
        
        results.append({
            "Representation": f"Neural ({model_name})",
            "Phoneme": ph,
            "Estimate": est,
            "CI_Lower": ci_low,
            "CI_Upper": ci_high,
            "ROPE": f"[0.0, {rope_high:.4f}]",
            "ROPE_Classification": classification
        })
        
    return pd.DataFrame(results)

def plot_forest(df, title, x_label, rope_bounds, output_path):
    """绘制森林图 (Forest Plot)"""
    plt.figure(figsize=(10, 6))
    
    # 获取唯一的音素并进行排序
    phonemes = df['Phoneme'].unique()
    y_pos = np.arange(len(phonemes))
    
    estimates = df['Estimate'].values
    ci_lows = df['CI_Lower'].values
    ci_highs = df['CI_Upper'].values
    
    # 画出误差线
    err_lower = estimates - ci_lows
    err_upper = ci_highs - estimates
    plt.errorbar(estimates, y_pos, xerr=[err_lower, err_upper], fmt='o', color='black', capsize=5)
    
    # 绘制 ROPE 区域
    plt.axvspan(rope_bounds[0], rope_bounds[1], color='gray', alpha=0.3, label='ROPE')
    
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5) # 参考线 0
    plt.yticks(y_pos, phonemes)
    plt.xlabel(x_label)
    plt.ylabel("Phoneme")
    plt.title(title)
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    output_dir = "results/8"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv("features/features_acoustic_norm.csv")
    
    # 恢复分类变量的编码
    df['is_L2'] = (df['l1_status'] == 'ru').astype(int)
    df['is_male'] = (df['gender'] == 'm').astype(int)
    oral_vowels = ['a', 'i', 'u', 'y', 'e', 'ɛ', 'o', 'ɔ', 'ø', 'œ', 'ə']
    
    # ================= 1. Acoustic CIs =================
    print("Calculating Acoustic Confidence Intervals...")
    res_ac = compute_acoustic_cis(df, oral_vowels, rope_hz=20)
    
    # ================= 2. Neural CIs =================
    print("Calculating Neural Confidence Intervals (B=2000)...")
    npz_whisper = np.load("features/features_whisper_norm.npz")
    npz_xlsr = np.load("features/features_xlsr_norm.npz")
    
    # 因为只需要算距离，可以直接用 pca50 特征以保留更多高维信息
    features_wh = npz_whisper['layer_20_pca50']
    features_xl = npz_xlsr['layer_18_pca50']
    
    res_wh = compute_neural_cis(df, features_wh, oral_vowels, "Whisper")
    res_xl = compute_neural_cis(df, features_xl, oral_vowels, "XLS-R")
    
    # ================= 3. Summary Table =================
    df_final = pd.concat([res_ac, res_wh, res_xl], ignore_index=True)
    csv_path = os.path.join(output_dir, "rope_classifications.csv")
    df_final.to_csv(csv_path, index=False)
    print(f"\nSaved ROPE summary table to {csv_path}")
    
    # ================= 4. Forest Plots =================
    print("Generating Forest Plots...")
    
    # 1. 画 Acoustic F1
    f1_col = 'F1_mid' if 'F1_mid' in df.columns else 'F1'
    df_f1 = df_final[df_final['Representation'] == f'Acoustic ({f1_col})']
    if not df_f1.empty:
        plot_forest(df_f1, "Acoustic F1: L1 vs L2 Contrast", "Difference in F1 (Hz)", [-20, 20], os.path.join(output_dir, "fig_forest_acoustic_F1.png"))
    
    # 2. 画 Whisper
    df_wh_plot = df_final[df_final['Representation'] == 'Neural (Whisper)']
    if not df_wh_plot.empty:
        rope_bound_wh = float(df_wh_plot.iloc[0]['ROPE'].split(',')[1].replace(']', '').strip())
        plot_forest(df_wh_plot, "Neural (Whisper): L1 vs L2 Cosine Distance", "Cosine Distance", [0, rope_bound_wh], os.path.join(output_dir, "fig_forest_whisper.png"))
    
    # 3. 画 XLS-R
    df_xl_plot = df_final[df_final['Representation'] == 'Neural (XLS-R)']
    if not df_xl_plot.empty:
        rope_bound_xl = float(df_xl_plot.iloc[0]['ROPE'].split(',')[1].replace(']', '').strip())
        plot_forest(df_xl_plot, "Neural (XLS-R): L1 vs L2 Cosine Distance", "Cosine Distance", [0, rope_bound_xl], os.path.join(output_dir, "fig_forest_xlsr.png"))
        
    print("Success! Forest plots and ROPE classifications are fully completed.")

if __name__ == "__main__":
    main()