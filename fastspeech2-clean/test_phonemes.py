#!/usr/bin/env python3
"""Test phoneme conversion to debug audio quality issues."""

import sys
from g2p_en import G2p

# ARPAbet to IPA mapping (same as in inference_prosody.py)
ARPA_TO_IPA = {
    # Vowels
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ə', 'AO': 'ɔ', 'AW': 'aw',
    'AY': 'aj', 'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'ej', 'IH': 'ɪ',
    'IY': 'iː', 'OW': 'ow', 'OY': 'ɔj', 'UH': 'ʊ', 'UW': 'uː',
    
    # Consonants
    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'F': 'f',
    'G': 'ɡ', 'HH': 'h', 'JH': 'dʒ', 'K': 'k', 'L': 'l',
    'M': 'm', 'N': 'n', 'NG': 'ŋ', 'P': 'p', 'R': 'ɹ',
    'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'V': 'v',
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
    
    # Special
    ' ': 'spn',
}

# Load phone vocabulary
phone_to_id = {}
with open('processed/phones.tsv', 'r', encoding='utf-8') as f:
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            phone, phone_id = parts
            phone_to_id[phone] = int(phone_id)

g2p = G2p()

def test_text(text):
    """Test phoneme conversion for a text."""
    print(f"\n{'='*60}")
    print(f"Text: \"{text}\"")
    print(f"{'='*60}")
    
    # Get ARPAbet phonemes
    arpa_phonemes = g2p(text)
    print(f"\n1. ARPAbet output: {' '.join(arpa_phonemes)}")
    
    # Convert to IPA
    ipa_phonemes = []
    phone_ids = []
    missing_phones = []
    
    for arpa in arpa_phonemes:
        arpa_clean = arpa.rstrip('012')  # Remove stress
        
        if arpa_clean == ' ':
            ipa = 'spn'
        elif arpa_clean in '.,!?;:-':
            ipa = 'spn'
        else:
            ipa = ARPA_TO_IPA.get(arpa_clean, arpa_clean.lower())
        
        ipa_phonemes.append(ipa)
        
        # Get phone ID
        if ipa in phone_to_id:
            phone_ids.append(str(phone_to_id[ipa]))
        else:
            phone_ids.append(f"MISSING({ipa})")
            missing_phones.append(ipa)
    
    print(f"2. IPA phonemes:   {' '.join(ipa_phonemes)}")
    print(f"3. Phone IDs:      {' '.join(phone_ids)}")
    
    if missing_phones:
        print(f"\n⚠ WARNING: Missing phones in vocabulary: {set(missing_phones)}")
        print("  These phones won't produce correct audio!")
    else:
        print("\n✓ All phonemes mapped successfully!")
    
    return len(missing_phones) == 0


if __name__ == "__main__":
    # Test common phrases
    test_cases = [
        "Hello world",
        "How are you today?",
        "This is a test of the speech synthesis system.",
    ]
    
    if len(sys.argv) > 1:
        # Test custom text
        text = ' '.join(sys.argv[1:])
        test_text(text)
    else:
        # Test all cases
        all_passed = True
        for text in test_cases:
            passed = test_text(text)
            all_passed = all_passed and passed
        
        print(f"\n{'='*60}")
        if all_passed:
            print("✓ All tests passed!")
        else:
            print("✗ Some tests failed - check missing phonemes above")
        print(f"{'='*60}\n")
