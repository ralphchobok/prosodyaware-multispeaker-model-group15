"""
Fix phone encoding in preprocessed files.

This script:
1. Collects all unique phones from .pt files
2. Creates phone vocabulary (phones.tsv)
3. Re-encodes phone text strings to IDs in all .pt files
"""

import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from collections import Counter

PROCESSED_DIR = Path("processed/")

def main():
    print("="*60)
    print("FIXING PHONE ENCODING")
    print("="*60)
    
    # Step 1: Collect all unique phones
    print("\nStep 1: Collecting phone vocabulary...")
    all_phones = []
    pt_files = list(PROCESSED_DIR.glob("**/*.pt"))
    
    for pt_file in tqdm(pt_files, desc="Scanning files"):
        data = torch.load(pt_file, weights_only=False)
        if 'encoded_text' in data and isinstance(data['encoded_text'], list):
            all_phones.extend(data['encoded_text'])
    
    phone_counts = Counter(all_phones)
    unique_phones = sorted(phone_counts.keys())
    
    print(f"\nFound {len(unique_phones)} unique phones")
    print(f"Total phone tokens: {len(all_phones):,}")
    print(f"\nMost common phones:")
    for phone, count in phone_counts.most_common(20):
        print(f"  {phone:>10} : {count:>8,}")
    
    # Step 2: Create phone-to-ID mapping
    phone_to_id = {phone: i for i, phone in enumerate(unique_phones)}
    
    # Save phones.tsv
    phone_df = pd.DataFrame([
        {'text': phone, 'phone_id': phone_id}
        for phone, phone_id in phone_to_id.items()
    ])
    phone_df.to_csv(PROCESSED_DIR / "phones.tsv", sep="\t", index=False)
    print(f"\n✓ Saved phone vocabulary to {PROCESSED_DIR / 'phones.tsv'}")
    
    # Step 3: Re-encode all files
    print("\nStep 2: Re-encoding phone text to IDs...")
    updated_count = 0
    error_count = 0
    
    for pt_file in tqdm(pt_files, desc="Encoding files"):
        try:
            data = torch.load(pt_file, weights_only=False)
            
            if 'encoded_text' in data and isinstance(data['encoded_text'], list):
                # Convert text phones to IDs
                phone_ids = [phone_to_id[phone] for phone in data['encoded_text']]
                data['encoded_text'] = torch.tensor(phone_ids, dtype=torch.long)
                
                # Save updated file
                torch.save(data, pt_file)
                updated_count += 1
        except Exception as e:
            print(f"\nError processing {pt_file}: {e}")
            error_count += 1
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Unique phones: {len(unique_phones)}")
    print(f"Files updated: {updated_count}")
    print(f"Errors: {error_count}")
    print(f"\n✓ Phone encoding complete!")
    print("="*60)

if __name__ == "__main__":
    main()
