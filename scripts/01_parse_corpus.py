import os
import re
import pandas as pd
import tgt

def main():
    # 1. Define paths
    data_dir = "data/ru-fr_interference/FRcorp_textgrids_only"
    metadata_path = "data/ru-fr_interference/metadata_RUFR.csv"
    rufrcorr_path = "data/ru-fr_interference/RUFRcorr.csv"
    output_path = "features/manifest.csv"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 2. Load Metadata (Speaker Info)
    print("Loading metadata...")
    metadata_df = pd.read_csv(metadata_path, sep=';')
    metadata_df = metadata_df[['spk', 'L1', 'Gender']]
    metadata_df.columns = ['speaker_id', 'l1_status', 'gender']
    
    # 3. Build Mapping Dictionary from RUFRcorr.csv
    # Mapping logic: File ID (from filename) -> {'sentence_id': '...', 'repetition': ...}
    print("Building file-to-sentence mapping dictionary...")
    corr_df = pd.read_csv(rufrcorr_path, sep=None, engine='python', encoding='utf-8-sig')
    file_mapping = {}
    
    for index, row in corr_df.iterrows():
        sentence_label = str(row['Word']).strip()
        # Loop through occ.1 to occ.6
        for rep_idx in range(1, 7):
            col_name = f'occ.{rep_idx}'
            if col_name in corr_df.columns and not pd.isna(row[col_name]):
                # The value in the cell is the File ID (e.g., 1, 18, 28)
                # Ensure it's cleanly parsed as a string without decimals
                file_id = str(int(row[col_name])) 
                file_mapping[file_id] = {
                    'sentence_id': sentence_label,
                    'repetition': rep_idx
                }

    # 4. Parse TextGrids
    records = []
    print("Parsing TextGrid files...")
    
    for speaker_folder in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_folder)
        if not os.path.isdir(speaker_path):
            continue
            
        speaker_id = speaker_folder
        
        for file in os.listdir(speaker_path):
            if file.endswith(".TextGrid"):
                # Use regex to extract the file ID, e.g., "FRcorp18" -> "18"
                match = re.search(r'FRcorp(\d+)', file)
                if not match:
                    continue
                
                file_id = match.group(1)
                
                # Look up the sentence and repetition from our dictionary
                if file_id in file_mapping:
                    sentence_id = file_mapping[file_id]['sentence_id']
                    repetition = file_mapping[file_id]['repetition']
                else:
                    sentence_id = "UNKNOWN"
                    repetition = -1
                
                # Read the actual TextGrid
                tg_path = os.path.join(speaker_path, file)
                tg = tgt.io.read_textgrid(tg_path)
                
                # Fetch the 'phones' tier directly based on your screenshot
                try:
                    phoneme_tier = tg.get_tier_by_name('phones')
                except ValueError:
                    # Defensive fallback just in case some files are named differently
                    phoneme_tier = tg.tiers[-1] 
                
                # 5. Extract Phonemes
                for interval in phoneme_tier:
                    label = interval.text.strip()
                    
                    # Filter out silence intervals
                    if label and label not in ['sil', 'sp', '']:
                        onset = interval.start_time
                        offset = interval.end_time
                        
                        records.append({
                            "speaker_id": speaker_id,
                            "sentence_id": sentence_id,     # e.g., "j'en chie"
                            "file_id":file_id,
                            "repetition": repetition,       # e.g., 2
                            "phoneme": label,               # e.g., "ʒ"
                            "onset": onset,
                            "offset": offset,
                            "duration": offset - onset
                        })

    # 6. Merge and Export
    print(f"Total phonemes extracted: {len(records)}")
    df = pd.DataFrame(records)
    
    print("Merging with metadata...")
    final_df = df.merge(metadata_df, on="speaker_id", how="left")
    
    final_df.to_csv(output_path, index=False)
    print(f"Success! Data saved to {output_path}")

if __name__ == "__main__":
    main()