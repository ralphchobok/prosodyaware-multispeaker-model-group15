#!/usr/bin/env python3
"""
Build phones.tsv from TextGrid files.
This is needed when preprocessing skips files that already exist and phones are already encoded as IDs.
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re

def extract_phones_from_textgrid(textgrid_path):
    """Extract all phones from a TextGrid file by parsing it directly."""
    phones = set()
    try:
        with open(textgrid_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the phones tier section
        # Look for: name = "phones" followed by intervals
        phones_tier_match = re.search(r'name = "phones".*?(?=item \[|$)', content, re.DOTALL)
        
        if phones_tier_match:
            phones_tier = phones_tier_match.group(0)
            
            # Extract all text values from intervals
            # Pattern: text = "phone"
            text_matches = re.findall(r'text = "([^"]*)"', phones_tier)
            
            for phone in text_matches:
                phone = phone.strip()
                if phone and phone != '':  # Skip empty intervals
                    phones.add(phone)
    except Exception as e:
        pass  # Skip files that can't be read
    
    return phones

def main():
    processed_dir = Path("processed")
    textgrid_dir = processed_dir / "textgrids"
    
    if not textgrid_dir.exists():
        print(f"ERROR: TextGrid directory not found: {textgrid_dir}")
        return
    
    # Collect all phones from TextGrid files
    vocab_phones = set()
    
    print("Collecting phones from TextGrid files...")
    textgrid_files = list(textgrid_dir.glob("**/*.TextGrid"))
    
    for tg_file in tqdm(textgrid_files, desc="Reading TextGrids"):
        phones = extract_phones_from_textgrid(tg_file)
        vocab_phones.update(phones)
    
    if not vocab_phones:
        print("ERROR: No phones found in TextGrid files!")
        return
    
    # Create vocabulary
    phone_vocab = sorted(list(vocab_phones))
    
    print(f"\nFound {len(phone_vocab)} unique phones")
    print("First 20 phones:", phone_vocab[:20])
    
    # Save to TSV
    phone_df = pd.DataFrame([
        {'text': phone, 'phone_id': i}
        for i, phone in enumerate(phone_vocab)
    ])
    
    output_file = processed_dir / "phones.tsv"
    phone_df.to_csv(output_file, sep="\t", index=False)
    
    print(f"\n✓ Saved vocabulary to {output_file}")
    print(f"  Total phones: {len(phone_vocab)}")

if __name__ == "__main__":
    main()
