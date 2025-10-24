import re
import sys
import pandas as pd
import numpy as np

# --- Conversion constants from your spec ---
GRAMS_PER_OZ = 28.3495
ML_PER_FL_OZ = 29.5735

UNIT_CONVERSIONS = {
    # Weight (Base: 'oz')
    'oz':    {'base': 'oz',     'factor': 1.0},
    'gram':  {'base': 'oz',     'factor': 1.0 / GRAMS_PER_OZ},
    'pound': {'base': 'oz',     'factor': 16.0},     # pounds -> ounces (value * 16)
    # Volume (Base: 'fl oz')
    'fl oz': {'base': 'fl oz',  'factor': 1.0},
    'ml':    {'base': 'fl oz',  'factor': 1.0 / ML_PER_FL_OZ},
    # Other (No conversion)
    'count': {'base': 'count',  'factor': 1.0},
}

# mapping for small spelled-out numbers to digits
WORDS_TO_NUM = {
    'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7,
    'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'thirteen':13,
    'fourteen':14, 'fifteen':15, 'sixteen':16, 'seventeen':17, 'eighteen':18,
    'nineteen':19, 'twenty':20
}

# Regex patterns to find pack counts: handles
# 12-pack, 12 pack, pack of 12, pack of twelve, 6pk, 6 pk, 3x pack, 3 x pack, three-pack
NUMERIC_PACK_PATTERNS = [
    r'(\d+)\s*[-]?\s*pack\b',         # 12-pack , 12 pack
    r'pack(?:\s+of)?\s+(\d+)\b',      # pack of 12, pack 12
    r'(\d+)\s*(?:pk)\b',              # 6pk, 6 pk
    r'(\d+)\s*[xX]\s*pack',           # 3xpack, 3 x pack
    r'(\d+)\s*[xX]\b',                # 3x, 3 x (rare)
]

WORD_PACK_PATTERNS = [
    r'(' + r'|'.join(WORDS_TO_NUM.keys()) + r')\s*[-]?\s*pack\b',   # twelve-pack
    r'pack(?:\s+of)?\s+(' + r'|'.join(WORDS_TO_NUM.keys()) + r')\b',# pack of twelve
]

def extract_pack_count(desc: str):
    """
    Try to extract a pack count X from description text.
    Returns integer X if found, otherwise None.
    """
    if not isinstance(desc, str):
        return None
    s = desc.lower()

    # try numeric patterns first
    for pat in NUMERIC_PACK_PATTERNS:
        m = re.search(pat, s)
        if m:
            try:
                val = int(m.group(1))
                if val > 0:
                    return val
            except Exception:
                pass

    # try spelled-out words
    for pat in WORD_PACK_PATTERNS:
        m = re.search(pat, s)
        if m:
            token = m.group(1).lower()
            if token in WORDS_TO_NUM:
                return WORDS_TO_NUM[token]

    # some entries write like 'six pack' without hyphen and without 'pack' trailing
    # handled above, but also catch patterns like '6 count' or '6 ct'
    m = re.search(r'(\d+)\s*(?:count|ct|cts)\b', s)
    if m:
        try:
            v = int(m.group(1))
            if v > 0:
                return v
        except:
            pass

    # fallback: look for patterns like 'pack of many' - we won't guess
    return None

def process_dataframe(df: pd.DataFrame):
    # Ensure expected columns exist
    required = {'desc_for_llm', 'value_after_conv', 'unit_normalized'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Normalize unit strings (lower-case, strip)
    df['unit_normalized_proc'] = df['unit_normalized'].astype(str).str.strip().str.lower()

    # Ensure numeric values
    df['value_after_conv_numeric'] = pd.to_numeric(df['value_after_conv'], errors='coerce')

    # Initialize new columns
    df['Count'] = np.nan
    df['oz'] = 0.0
    df['fl_oz'] = 0.0

    # First: handle count rows explicitly
    mask_count_unit = df['unit_normalized_proc'] == 'count'
    df.loc[mask_count_unit, 'Count'] = df.loc[mask_count_unit, 'value_after_conv_numeric']
    df.loc[mask_count_unit, ['oz', 'fl_oz']] = 0.0

    # Now handle oz, gram, pound -> oz; ml, fl oz -> fl_oz
    for idx, row in df[~mask_count_unit].iterrows():
        unit = row['unit_normalized_proc']
        val = row['value_after_conv_numeric']
        if pd.isna(val):
            # can't convert, leave zeros / NaN
            continue

        if unit in ('oz', 'gram', 'pound'):
            # convert to ounces (oz)
            if unit == 'oz':
                oz_val = float(val)
            else:
                # use UNIT_CONVERSIONS factors
                factor = UNIT_CONVERSIONS.get(unit, {}).get('factor', None)
                if factor is None:
                    oz_val = 0.0
                else:
                    oz_val = float(val) * factor
            df.at[idx, 'oz'] = oz_val
            df.at[idx, 'fl_oz'] = 0.0

        elif unit in ('fl oz', 'ml'):
            # convert to fluid ounces
            if unit == 'fl oz':
                floz_val = float(val)
            else:
                factor = UNIT_CONVERSIONS.get(unit, {}).get('factor', None)
                if factor is None:
                    floz_val = 0.0
                else:
                    floz_val = float(val) * factor
            df.at[idx, 'fl_oz'] = floz_val
            df.at[idx, 'oz'] = 0.0

        else:
            # unknown or other units: keep zeros
            df.at[idx, 'oz'] = 0.0
            df.at[idx, 'fl_oz'] = 0.0

    # For non-count rows that are in the set (oz, fl oz, gram, ml, pound) we attempt to extract pack counts
    packable_units = {'oz', 'fl oz', 'gram', 'ml', 'pound'}
    mask_packable = df['unit_normalized_proc'].isin(packable_units)
    for idx, row in df[mask_packable].iterrows():
        desc = row.get('desc_for_llm', '')
        extracted = extract_pack_count(desc)
        if extracted is not None:
            df.at[idx, 'Count'] = extracted
        else:
            # if pack not found, set count = 1
            # BUT only if Count not set already
            if pd.isna(df.at[idx, 'Count']):
                df.at[idx, 'Count'] = 1

    # For any remaining NaN Count values (e.g., some unexpected units), fill with 0 or keep NaN as you prefer.
    df['Count'] = df['Count'].fillna(0).astype(int)

    # Drop helper columns
    df = df.drop(columns=['unit_normalized_proc', 'value_after_conv_numeric'])

    return df

def main(input_csv, output_csv=None):
    df = pd.read_csv(input_csv)
    out = process_dataframe(df)
    if output_csv is None:
        output_csv = input_csv.rsplit('.', 1)[0] + '_with_counts.csv'
    out.to_csv(output_csv, index=False)
    print(f"Saved output to {output_csv}")

if __name__ == '__main__':
    ## PUT PATHS HERE TO CSVs from STAGE 1 PREPROCESSING
    inp = '/kaggle/input/amazon-ml-dataset-csv/preprocessed/test_norm.csv'
    outp = '/kaggle/working/test_split_final.csv'
    main(inp, outp)
