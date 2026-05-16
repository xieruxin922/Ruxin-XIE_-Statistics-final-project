import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy.stats import chi2 

warnings.simplefilter('ignore', ConvergenceWarning)

def calculate_r2_nakagawa(mdf):
    """计算 LME 的 Marginal R2 和 Conditional R2 (Nakagawa & Schielzeth, 2013)"""
    try:
        # 🚀 核心修复：直接使用固定效应的设计矩阵 (exog) 乘以固定效应系数 (fe_params)
        # 这样只计算纯粹的固定效应预测值 (Fixed effects predictor)，
        # 彻底绕过 statsmodels 提取随机效应时遇到的 Singular 崩溃陷阱！
        fixed_predictions = np.dot(mdf.model.exog, mdf.fe_params)
        var_f = np.var(fixed_predictions)
        
        # 随机截距方差 (Random intercept variance)
        var_u = mdf.cov_re.iloc[0, 0]
        
        # 残差方差 (Residual variance)
        var_e = mdf.scale
        
        total_var = var_f + var_u + var_e
        r2_m = var_f / total_var if total_var > 0 else np.nan
        r2_c = (var_f + var_u) / total_var if total_var > 0 else np.nan
        
        return r2_m, r2_c
        
    except Exception as e:
        print(f"  [Warning] Could not calculate R2: {e}")
        return np.nan, np.nan
    

def map_vowel_height(phoneme):
    """将法语口腔元音映射为舌位高低 (Phonological Context)"""
    high = ['i', 'y', 'u']
    mid = ['e', 'ø', 'o', 'ɛ', 'œ', 'ɔ', 'ə']
    low = ['a']
    if phoneme in high: return 'high'
    elif phoneme in mid: return 'mid'
    elif phoneme in low: return 'low'
    else: return 'unknown'

def fit_lme_hierarchy(df, target_col):
    """拟合 5 个层级的模型，并进行似然比检验 (LRT)"""
    results = []
    models = {}
    
    formulas = {
        "1_Null": f"{target_col} ~ 1",
        "2_Main": f"{target_col} ~ is_L2 + is_male",
        "3_Full": f"{target_col} ~ is_L2 * is_male",
        "4_Extended": f"{target_col} ~ is_L2 * is_male + C(vowel_height)"
    }
    
    prev_mdf = None
    prev_name = None
    
    print(f"\n--- Fitting models for {target_col} ---")
    
    for name, formula in formulas.items():
        try:
            # 必须使用 ML (reml=False)
            md = smf.mixedlm(formula, df, groups=df["speaker_id"])
            mdf = md.fit(method='lbfgs', reml=False)
            models[name] = mdf
            
            var_u = mdf.cov_re.iloc[0, 0]
            var_e = mdf.scale
            icc = var_u / (var_u + var_e) if (var_u + var_e) > 0 else np.nan
            r2_m, r2_c = calculate_r2_nakagawa(mdf)
            
            # 🚀 核心修复：纯手工计算似然比检验 (LRT)，绕过官方的 bug
            lrt_p = np.nan
            if prev_mdf is not None:
                # 提取对数似然值 (Log-Likelihood)
                lr_stat = -2 * (prev_mdf.llf - mdf.llf)
                # 计算自由度差异 (参数数量之差)
                df_diff = len(mdf.params) - len(prev_mdf.params)
                if df_diff > 0:
                    lrt_p = chi2.sf(lr_stat, df_diff) # 使用卡方分布求 p 值
                
            results.append({
                "Target": target_col,
                "Model": name,
                "AIC": mdf.aic,
                "BIC": mdf.bic,
                "LogLike": mdf.llf,
                "ICC": icc,
                "Marginal R2": r2_m,
                "Conditional R2": r2_c,
                f"LRT p-value (vs {prev_name})": lrt_p
            })
            
            prev_mdf = mdf
            prev_name = name
        except Exception as e:
            print(f"Failed to fit {name} for {target_col}: {e}")

    # 5. 随机斜率陷阱
    name = "5_RandomSlope"
    try:
        md = smf.mixedlm(formulas["4_Extended"], df, groups=df["speaker_id"], re_formula="~is_L2")
        mdf = md.fit(method='lbfgs', reml=False)
        print("  -> SURPRISE: Random slope model converged!")
    except Exception as e:
        print("  -> EXPECTED FAILURE: Random slope for L1 status failed.")
        results.append({
            "Target": target_col,
            "Model": name,
            "Notes": "Failed intentionally. L1 is between-speaker (variance=0)."
        })

    return pd.DataFrame(results), models.get("4_Extended")


def main():
    output_dir = "results/7"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading datasets...")
    df = pd.read_csv("features/features_acoustic_norm.csv")
    
    # 1. 数据预处理与编码 (Coding)
    # L2: 1 if L2 (ru), 0 if L1 (fr)
    df['is_L2'] = (df['l1_status'] == 'ru').astype(int)
    # Male: 1 if male, 0 if female
    df['is_male'] = (df['gender'] == 'm').astype(int)
    # Vowel Height
    df['vowel_height'] = df['phoneme'].apply(map_vowel_height)
    
    # 过滤掉不合法的元音
    df = df[df['vowel_height'] != 'unknown'].copy()
    valid_idx = df.index.values

    # 2. 将神经网络前 5 个主成分 (PCs) 挂载到 DataFrame 上
    npz_whisper = np.load("features/features_whisper_norm.npz")
    npz_xlsr = np.load("features/features_xlsr_norm.npz")
    
    # 我们知道 pca50 的前 5 列就是前 5 个主成分
    whisper_pc5 = npz_whisper['layer_20_pca50'][valid_idx, :5]
    xlsr_pc5 = npz_xlsr['layer_18_pca50'][valid_idx, :5]
    
    for i in range(5):
        df[f'Wh_PC{i+1}'] = whisper_pc5[:, i]
        df[f'Xl_PC{i+1}'] = xlsr_pc5[:, i]

    # 3. 执行所有的 LME
    all_results = []
    best_models = {}
    
    # 定义我们要检验的所有目标变量
    targets = ['F1_mid_norm', 'F2_mid_norm'] + [f'Wh_PC{i+1}' for i in range(5)] + [f'Xl_PC{i+1}' for i in range(5)]
    
    for target in targets:
        # 去掉缺失值，否则 statsmodels 会报错
        df_target = df.dropna(subset=[target, 'is_L2', 'is_male', 'vowel_height']).copy()
        res_df, best_model = fit_lme_hierarchy(df_target, target)
        all_results.append(res_df)
        if best_model is not None:
            best_models[target] = best_model

    # 合并并保存所有的模型比较结果
    final_report = pd.concat(all_results, ignore_index=True)
    report_path = os.path.join(output_dir, "lme_hierarchy_results.csv")
    final_report.to_csv(report_path, index=False)
    print(f"\nSaved massive LME report to {report_path}")

    # 4. 提取用于回答 7.4 (Comparing Representation Types) 的最终表格
    print("\nExtracting R2 comparison for the Extended models...")
    comp_results = []
    
    # 提取 F1, Whisper PC1, XLS-R PC1 作为代表进行横向对比
    for target in ['F1_mid_norm', 'Wh_PC1', 'Xl_PC1']:
        if target in best_models:
            mdf = best_models[target]
            r2_m, r2_c = calculate_r2_nakagawa(mdf)
            comp_results.append({
                "Representation": target,
                "Marginal R2 (Fixed Effects)": round(r2_m, 4),
                "Conditional R2 (Total)": round(r2_c, 4)
            })
            
    pd.DataFrame(comp_results).to_csv(os.path.join(output_dir, "lme_representation_comparison.csv"), index=False)
    print("Success! LME pipeline complete.")

if __name__ == "__main__":
    main()