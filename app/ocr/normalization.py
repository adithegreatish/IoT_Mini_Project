"""Indian vehicle registration plate text normalization and format validation.

Supports:
1. Standard Indian plates (CMVR Rule 50 & 51): State (2 letters) + District (1-2 digits) + Series (0-3 letters) + Number (1-4 digits)
2. BH Series (Bharat Series, 2021+): Year (2 digits) + BH + Number (4 digits) + Series (1-2 letters)
3. Trade Certificate (TC) & Commercial plates
4. Position-aware character disambiguation between visually confusable glyphs (O<->0, I/L<->1, Z<->2, S<->5, B<->8)
5. Exclusion of HSRP security features (IND emblem, laser-etched PIN) and frame dealership markings
"""

import re
from typing import Optional, Tuple

# All 28 Indian States and 8 Union Territories with official abbreviations and common variants
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "CT", "DD", "DH", "DL", "DN",
    "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD",
    "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ",
    "SK", "TN", "TS", "TR", "UA", "UK", "UP", "WB"
}

# Standard CMVR registration regex:
# Group 1: 2-letter State code
# Group 2: 1-2 digit District RTO code
# Group 3: 0-3 letter series code
# Group 4: 1-4 digit unique vehicle number
STANDARD_PLATE_REGEX = re.compile(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{1,4})$")

# Bharat Series (BH) regex (e.g., 22BH1234AA)
BH_PLATE_REGEX = re.compile(r"^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$")

# Trade Certificate (TC) regex (e.g., MH12TC1234)
TC_PLATE_REGEX = re.compile(r"^([A-Z]{2})([0-9]{1,2})TC([0-9]{1,4})$")

# Known dealer frame / manufacturer brand words to strip if detected on license plate borders
FRAME_BRAND_TOKENS = {
    "MARUTI", "SUZUKI", "JZUKI", "HYUNDAI", "HONDA", "TOYOTA", "TATA",
    "MAHINDRA", "FORD", "NISSAN", "RENAULT", "VOLKSWAGEN", "SKODA", "KIA",
    "MOTORS", "INDIA", "BHARAT", "HERO", "BAJAJ", "TVS", "YAMAHA", "ROYAL", "ENFIELD"
}

# Character disambiguation dictionaries
CHAR_TO_DIGIT = {
    "O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"
}
DIGIT_TO_CHAR = {
    "0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"
}


def disambiguate_registration_string(clean: str) -> Tuple[str, bool, Optional[str]]:
    """
    Performs deterministic, slot-based disambiguation on cleaned candidate alphanumeric string.
    Only replaces confusable characters if the statutory position strictly requires a digit or letter.
    Never invents missing characters.
    """
    if len(clean) < 6:
        return clean, False, "Registration string too short (<6 characters)"

    # 1. Check BH Series
    bh_m = BH_PLATE_REGEX.match(clean)
    if bh_m:
        formatted = f"{bh_m.group(1)} BH {bh_m.group(2)} {bh_m.group(3)}"
        return formatted, True, None

    # 2. Check TC Series
    tc_m = TC_PLATE_REGEX.match(clean)
    if tc_m:
        state = tc_m.group(1)
        if state in INDIAN_STATE_CODES:
            return f"{state} {tc_m.group(2)} TC {tc_m.group(3)}", True, None

    # 3. Check Standard format directly
    m = STANDARD_PLATE_REGEX.match(clean)
    if m:
        state = m.group(1)
        if state in INDIAN_STATE_CODES:
            series = m.group(3)
            formatted = f"{state} {m.group(2)}" + (f" {series}" if series else "") + f" {m.group(4)}"
            return formatted, True, None

    # 4. Slot-based disambiguation
    chars = list(clean)

    # Slot 0-1: State code (MUST BE LETTERS)
    for i in (0, 1):
        if chars[i] in DIGIT_TO_CHAR:
            chars[i] = DIGIT_TO_CHAR[chars[i]]

    state_candidate = "".join(chars[:2])
    if state_candidate not in INDIAN_STATE_CODES:
        return clean, False, f"Unrecognized state code '{state_candidate}'"

    rem = chars[2:]
    if len(rem) < 4:
        return clean, False, "Insufficient characters following state code"

    # Evaluate district length (2 digits vs 1 digit)
    for dist_len in (2, 1):
        test_rem = list(rem)
        valid_dist = True
        for i in range(dist_len):
            if test_rem[i] in CHAR_TO_DIGIT:
                test_rem[i] = CHAR_TO_DIGIT[test_rem[i]]
            elif not test_rem[i].isdigit():
                valid_dist = False
                break
        if not valid_dist:
            continue

        dist_str = "".join(test_rem[:dist_len])
        after_dist = test_rem[dist_len:]
        if len(after_dist) < 1:
            continue

        # Last 1-4 characters MUST BE DIGITS (Unique Number)
        # Remaining characters between district and number MUST BE LETTERS (Series)
        for num_len in (4, 3, 2, 1):
            if len(after_dist) < num_len:
                continue
            series_chars = list(after_dist[:-num_len])
            num_chars = list(after_dist[-num_len:])

            if len(series_chars) > 3:
                continue

            # Verify and fix series letters
            valid_series = True
            for k in range(len(series_chars)):
                if series_chars[k] in DIGIT_TO_CHAR:
                    series_chars[k] = DIGIT_TO_CHAR[series_chars[k]]
                elif not series_chars[k].isalpha():
                    valid_series = False
                    break
            if not valid_series:
                continue

            # Verify and fix number digits
            valid_num = True
            for k in range(len(num_chars)):
                if num_chars[k] in CHAR_TO_DIGIT:
                    num_chars[k] = CHAR_TO_DIGIT[num_chars[k]]
                elif not num_chars[k].isdigit():
                    valid_num = False
                    break
            if not valid_num:
                continue

            series_str = "".join(series_chars)
            num_str = "".join(num_chars)
            formatted = f"{state_candidate} {dist_str}" + (f" {series_str}" if series_str else "") + f" {num_str}"
            return formatted, True, None

    return clean, False, "Does not match standard Indian registration syntax"


def normalize_plate_text(raw_text: Optional[str]) -> Tuple[str, bool, Optional[str]]:
    """
    Cleans raw OCR string, strips unwanted symbols and HSRP emblems, validates Indian registration format.
    Returns: (normalized_text, is_valid_format, error_reason)
    """
    if not raw_text or not raw_text.strip():
        return "UNREADABLE", False, "Empty OCR output"

    # Tokenize by whitespace to filter individual frame words
    tokens = [t.strip().upper() for t in raw_text.split() if t.strip()]
    filtered_tokens = []
    for tok in tokens:
        clean_tok = re.sub(r"[^A-Za-z0-9]", "", tok)
        if not clean_tok:
            continue
        # Strip standalone security emblems and frame brands
        if clean_tok in ("IND", "INDIA") or clean_tok in FRAME_BRAND_TOKENS:
            continue
        # Strip 10-12 character laser PINs (e.g. AA2013376292)
        if len(clean_tok) >= 10 and re.match(r"^[A-Z]{2}[0-9]{8,10}$", clean_tok):
            continue
        filtered_tokens.append(clean_tok)

    if not filtered_tokens:
        return "UNREADABLE", False, "No registration characters identified"

    clean = "".join(filtered_tokens)

    # Discard leading or trailing 'IND' if glued to characters
    if clean.startswith("IND") and len(clean) > 8:
        clean = clean[3:]
    if clean.endswith("IND") and len(clean) > 8:
        clean = clean[:-3]

    return disambiguate_registration_string(clean)
