import os
import yaml
import torch
import librosa
import numpy as np
import pandas as pd
from transformers import WhisperModel, WhisperFeatureExtractor

def main():
    # 1. Load Parameters
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["whisper"]

    model_id = params["model_id"]
    target_layers = params["target_layers"]
    target_sr = params["sampling_rate"]

    # 2. Setup Device (自动检测是否有显卡可用，否则用 CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on {device}...")

    # Load HuggingFace Model and Feature Extractor
    extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    # 只需要 encoder，不用加载 decoder，节省内存
    model = WhisperModel.from_pretrained(model_id).to(device)
    model.eval() # 设置为评估模式

    # 3. Load Manifest
    df = pd.read_csv("features/manifest.csv")
    # df = df.head(10)
    # 记录原始索引，保证导出的向量顺序与 manifest.csv 的行一一对应
    df['orig_idx'] = df.index 
    
    # 准备一个字典来存放不同层的高维向量矩阵
    # 形状为: (音素总数, 1024) -> Whisper medium 维度是 1024
    d_model = model.config.d_model
    embeddings = {layer: np.zeros((len(df), d_model), dtype=np.float32) for layer in target_layers}

    # 4. Process by Audio File (和上一步一样的加速策略，整条音频只推理一次)
    data_dir = "data/ru-fr_interference/FRcorp_textgrids_only"
    grouped = df.groupby(['speaker_id', 'file_id'])
    
    # Whisper 编码器输出帧率：1帧 = 0.02秒 (50 Hz)
    FRAME_RATE = 50.0 

    for (speaker_id, file_id), group_df in grouped:
        speaker_dir = os.path.join(data_dir, str(speaker_id))
        wav_path = None
        if os.path.exists(speaker_dir):
            for f in os.listdir(speaker_dir):
                if f.endswith(f"FRcorp{file_id}.wav"):
                    wav_path = os.path.join(speaker_dir, f)
                    break
                    
        if not wav_path:
            continue
            
        # 4.1 Load audio and resample to 16kHz
        audio, _ = librosa.load(wav_path, sr=target_sr)
        
        # 4.2 Extract log-mel spectrogram and pass through encoder
        inputs = extractor(audio, return_tensors="pt", sampling_rate=target_sr).to(device)
        
        with torch.no_grad():
            # output_hidden_states=True 才会返回每一层的向量
            outputs = model.encoder(inputs.input_features, output_hidden_states=True)
            # hidden_states 是一个 tuple: (embedding层, 第1层, 第2层, ..., 第24层)
            hidden_states = outputs.hidden_states 

        # 4.3 Map phoneme timestamps to encoder frames
        for _, row in group_df.iterrows():
            idx = row['orig_idx']
            onset = row['onset']
            offset = row['offset']
            
            # 计算包含这个音素的帧的起止索引
            start_frame = int(onset * FRAME_RATE)
            end_frame = int(offset * FRAME_RATE)
            
            # 防御机制：如果音素极短，导致 start 和 end 落入同一帧，强制取那一帧
            if start_frame == end_frame:
                end_frame += 1
                
            for layer in target_layers:
                # 提取目标层的数据并放到 CPU 上
                # 形状为 (batch_size=1, sequence_length, hidden_size=1024)
                layer_hs = hidden_states[layer][0].cpu().numpy()
                
                # 防止索引越界 (比如音频末尾的静音段由于四舍五入超出界限)
                safe_end = min(end_frame, layer_hs.shape[0])
                safe_start = min(start_frame, layer_hs.shape[0] - 1)
                if safe_start >= safe_end:
                    safe_end = safe_start + 1
                
                # 公式 (1)：Average-pool across time T
                token_emb = np.mean(layer_hs[safe_start:safe_end, :], axis=0)
                
                # 存入大矩阵对应的行
                embeddings[layer][idx] = token_emb
                
    # 5. Export to .npz
    # 将包含多个层的矩阵保存为一个 .npz 压缩包
    output_dict = {f"layer_{layer}": embeddings[layer] for layer in target_layers}
    np.savez("features/features_whisper.npz", **output_dict)
    print("Success! Neural representations saved to features/features_whisper.npz")

if __name__ == "__main__":
    main()