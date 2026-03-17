#!/usr/bin/env python3
"""Debug phoneme conversion to check for vocabulary mismatches."""

import sys
from predict import convert_english_to_phonemes, convert_phones_to_tokens

def load_phone_vocab():
    """Load phone vocabulary from processed/phones.tsv"""
    phone_to_id = {}
    with open('processed/phones.tsv', 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                phone, phone_id = parts
                phone_to_id[phone] = int(phone_id)
    return phone_to_id

def debug_text_conversion(text, phone_to_id):
    """Debug the full text-to-phoneme conversion pipeline."""
    print(f"\n{'='*70}")
    print(f"Text: \"{text}\"")
    print(f"{'='*70}\n")
    
    # Step 1: Convert to IPA
    ipa_text = convert_english_to_phonemes(text)
    print(f"Step 1 - IPA conversion:")
    print(f"  {ipa_text}")
    print()
    
    # Step 2: Convert to tokens
    phone_ids, stress_ids, phonemes = convert_phones_to_tokens(phone_to_id, ipa_text)
    
    print(f"Step 2 - Token conversion:")
    print(f"  Phonemes: {phonemes}")
    print(f"  Phone IDs: {phone_ids}")
    print(f"  Stress IDs: {stress_ids}")
    print()
    
    # Step 3: Check for issues
    missing_phonemes = []
    for i, phone in enumerate(phonemes):
        if phone not in phone_to_id:
            missing_phonemes.append((i, phone))
    
    if missing_phonemes:
        print("❌ ERRORS FOUND:")
        for idx, phone in missing_phonemes:
            print(f"  Position {idx}: Phoneme '{phone}' NOT IN VOCABULARY")
        print()
    else:
        print("✓ All phonemes valid!")
        print()
    
    # Step 4: Show detailed mapping
    print("Detailed phoneme mapping:")
    for i, (phone, phone_id, stress) in enumerate(zip(phonemes, phone_ids, stress_ids)):
        stress_label = {1: 'unstressed', 2: 'primary', 3: 'secondary'}.get(stress, 'unknown')
        print(f"  {i:2d}. '{phone:4s}' → ID {phone_id:3d}  (stress: {stress_label})")
    
    return len(missing_phonemes) == 0

def main():
    phone_to_id = load_phone_vocab()
    print(f"Loaded {len(phone_to_id)} phonemes from vocabulary\n")
    
    # Test sentences
    test_sentences = [
        "Hello world",
        "This is a test",
        "How are you today?",
        "The quick brown fox jumps over the lazy dog",
    ]
    
    if len(sys.argv) > 1:
        # Use custom text from command line
        text = ' '.join(sys.argv[1:])
        debug_text_conversion(text, phone_to_id)
    else:
        # Test all default sentences
        all_passed = True
        for text in test_sentences:
            passed = debug_text_conversion(text, phone_to_id)
            all_passed = all_passed and passed
        
        print(f"\n{'='*70}")
        if all_passed:
            print("✓ ALL TESTS PASSED - Phoneme mapping is correct!")
        else:
            print("✗ SOME TESTS FAILED - Check phoneme mapping!")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
