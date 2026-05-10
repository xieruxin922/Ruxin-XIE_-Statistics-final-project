import os
import yaml
import torch
import librosa
import numpy as np
import pandas as pd
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

def main():
    # 1. Load Parameters
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["xlsr"]

    model_id = params["model_id"]
    target_layers = params["target_layers"]
    target_sr = params["sampling_rate"]

    # 2. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on {device}...")

    # Load HuggingFace Model (注意这里用的是 Wav2Vec2 系列类)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2Model.from_pretrained(model_id).to(device)
    model.eval()

    # 3. Load Manifest and initialize storage
    df = pd.read_csv("features/manifest.csv")
    # df = df.head(10)
    df['orig_idx'] = df.index 
    
    # Wav2Vec2-large 的 hidden_size 是 1024
    d_model = model.config.hidden_size 
    embeddings = {layer: np.zeros((len(df), d_model), dtype=np.float32) for layer in target_layers}

    # 4. Process by Audio File
    data_dir = "data/ru-fr_interference/FRcorp_textgrids_only"
    grouped = df.groupby(['speaker_id', 'file_id'])
    
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
            
        # 4.1 Load audio
        audio, _ = librosa.load(wav_path, sr=target_sr)
        
        # 4.2 XLS-R 直接输入原始波形波 (input_values)
        inputs = extractor(audio, return_tensors="pt", sampling_rate=target_sr).to(device)
        
        with torch.no_grad():
            outputs = model(inputs.input_values, output_hidden_states=True)
            # hidden_states: tuple, 索引 0 是 CNN 输出, 1-24 是 Transformer 层
            hidden_states = outputs.hidden_states 

        # 4.3 Map phoneme timestamps
        for _, row in group_df.iterrows():
            idx = row['orig_idx']
            onset = row['onset']
            offset = row['offset']
            
            start_frame = int(onset * FRAME_RATE)
            end_frame = int(offset * FRAME_RATE)
            
            if start_frame == end_frame:
                end_frame += 1
                
            for layer in target_layers:
                layer_hs = hidden_states[layer][0].cpu().numpy()
                
                safe_end = min(end_frame, layer_hs.shape[0])
                safe_start = min(start_frame, layer_hs.shape[0] - 1)
                if safe_start >= safe_end:
                    safe_end = safe_start + 1
                
                # Equation (1): Average-pool
                token_emb = np.mean(layer_hs[safe_start:safe_end, :], axis=0)
                embeddings[layer][idx] = token_emb
                
    # 5. Export to .npz
    output_dict = {f"layer_{layer}": embeddings[layer] for layer in target_layers}
    np.savez("features/features_xlsr.npz", **output_dict)
    print("Success! XLS-R neural representations saved to features/features_xlsr.npz")

if __name__ == "__main__":
    main()