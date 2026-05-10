import os
import yaml
import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call

def main():
    # Load Parameters
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["acoustics"]

    # 1. Define paths
    data_dir = "data/ru-fr_interference/FRcorp_textgrids_only"
    manifest_path = "features/manifest.csv"
    output_path = "features/features_acoustic.csv"
    
    print("Loading manifest...")
    df = pd.read_csv(manifest_path)
    
    extracted_features = []
    
    # To optimize, we group by speaker and file_id so we only load each WAV file ONCE
    grouped = df.groupby(['speaker_id', 'file_id'])
    
    print(f"Starting acoustic extraction for {len(grouped)} audio files...")
    
    for (speaker_id, file_id), group_df in grouped:
        # Construct the directory path for this speaker
        speaker_dir = os.path.join(data_dir, speaker_id)
        
        # Find the corresponding .wav file
        wav_path = None
        if os.path.exists(speaker_dir):
            for f in os.listdir(speaker_dir):
                if f.endswith(f"FRcorp{file_id}.wav"):
                    wav_path = os.path.join(speaker_dir, f)
                    break
        
        if not wav_path or not os.path.exists(wav_path):
            print(f"Warning: WAV file not found for Speaker {speaker_id}, File {file_id}")
            continue
            
        # 2. Load Audio in Praat (Parselmouth)
        snd = parselmouth.Sound(wav_path)
        
        # Determine Gender for LPC Parameters
        # Rule: max_formant = 5000Hz for female, 4500Hz for male
        gender = group_df['gender'].iloc[0].lower()
        max_formant = params['max_formant_female'] if gender == 'f' else params['max_formant_male']
        
        # Pre-compute Formant and Pitch objects for the WHOLE audio file
        formant_obj = snd.to_formant_burg(max_number_of_formants=params['n_formants'], maximum_formant=max_formant)
        pitch_obj = snd.to_pitch()
        
        # 3. Iterate through every phoneme in this WAV file
        for _, row in group_df.iterrows():
            phoneme = str(row['phoneme']).strip()
            onset = row['onset']
            offset = row['offset']
            duration_ms = row['duration'] * 1000  # Convert to ms
            
            midpoint = onset + (row['duration'] / 2)
            
            # --- Initialize features with NaN ---
            f1_mid, f2_mid, f3_mid, mean_f0, scg = np.nan, np.nan, np.nan, np.nan, np.nan
            f1_25, f2_25, f3_25 = np.nan, np.nan, np.nan
            f1_75, f2_75, f3_75 = np.nan, np.nan, np.nan
            
            # Simple Phonetic Classification (You may need to adjust these lists based on your specific IPA alphabet)
            is_vowel = phoneme in ['a', 'i', 'u', 'y', 'e', 'ɛ', 'o', 'ɔ', 'ø', 'œ', 'ə', 'ɑ̃', 'ɛ̃', 'ɔ̃', 'œ̃']
            is_voiced = is_vowel or phoneme in ['b', 'd', 'g', 'v', 'z', 'ʒ', 'ʁ', 'l', 'm', 'n', 'ɲ', 'w', 'j', 'ɥ']
            is_fricative = phoneme in ['f', 'v', 's', 'z', 'ʃ', 'ʒ', 'ʁ']
            
            # --- Extract Formants (F1, F2 for all; F3 for vowels) ---
            try:
                f1_mid = formant_obj.get_value_at_time(1, midpoint)
                f2_mid = formant_obj.get_value_at_time(2, midpoint)
                if is_vowel:
                    f3_mid = formant_obj.get_value_at_time(3, midpoint)
                    
                    # Long vowel trajectory logic (> 80 ms)
                    if duration_ms > params['long_vowel_threshold_ms']:
                        t_25 = onset + (row['duration'] * 0.25)
                        t_75 = onset + (row['duration'] * 0.75)
                        f1_25 = formant_obj.get_value_at_time(1, t_25)
                        f2_25 = formant_obj.get_value_at_time(2, t_25)
                        f3_25 = formant_obj.get_value_at_time(3, t_25)
                        f1_75 = formant_obj.get_value_at_time(1, t_75)
                        f2_75 = formant_obj.get_value_at_time(2, t_75)
                        f3_75 = formant_obj.get_value_at_time(3, t_75)
            except Exception:
                pass # Formant tracking failure
                
            # --- Extract f0 (Voiced only) ---
            if is_voiced:
                try:
                    # Get mean f0 over the duration
                    f0_values = []
                    for t in np.arange(onset, offset, params['f0_time_step']):
                        val = pitch_obj.get_value_at_time(t)
                        if not np.isnan(val):
                            f0_values.append(val)
                    if f0_values:
                        mean_f0 = np.mean(f0_values)
                except Exception:
                    pass
                    
            # --- Extract Spectral Center of Gravity (Fricatives only) ---
            if is_fricative:
                try:
                    # Slice the audio just for the fricative duration
                    snd_slice = snd.extract_part(from_time=onset, to_time=offset)
                    spectrum = snd_slice.to_spectrum()
                    scg = spectrum.get_center_of_gravity(params['scg_power'])
                except Exception:
                    pass
            
            # Store all features
            feature_dict = row.to_dict()
            feature_dict.update({
                'duration_ms': duration_ms,
                'F1_mid': f1_mid, 'F2_mid': f2_mid, 'F3_mid': f3_mid,
                'F1_25': f1_25, 'F2_25': f2_25, 'F3_25': f3_25,
                'F1_75': f1_75, 'F2_75': f2_75, 'F3_75': f3_75,
                'f0_mean': mean_f0,
                'SCG': scg
            })
            extracted_features.append(feature_dict)

    # 4. Create DataFrame and Calculate Missing Values Report
    print("\nFeature extraction complete. Handling missing values...")
    final_df = pd.DataFrame(extracted_features)
    
    # Requirement: Report proportion of missing values per phoneme class and group
    missing_report = final_df.groupby(['l1_status', 'gender','phoneme'])[['F1_mid', 'F2_mid', 'F3_mid', 'F1_25', 'F2_25', 'F3_25', 'F1_75', 'F2_75', 'F3_75', 'f0_mean', 'SCG']].apply(lambda x: x.isna().mean())
    missing_report.to_csv("features/missing_values_report.csv")
    print("Missing values report saved to features/missing_values_report.csv")
    
    # 5. Export to CSV
    final_df.to_csv(output_path, index=False)
    print(f"Success! Acoustic features saved to {output_path}")

if __name__ == "__main__":
    main()