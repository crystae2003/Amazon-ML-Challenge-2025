#!/usr/bin/env python3
"""
normalize_units.py

Usage:
    python normalize_units.py

Reads train.csv and test.csv in current directory, normalizes units and values
according to your specification.
"""
import os
import re
import math
import json
from typing import Tuple, Optional
import numpy as np
import pandas as pd

# Optional: OpenAI fallback if user provides API key
USE_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
if USE_OPENAI:
    try:
        import openai
        openai.api_key = os.environ.get("OPENAI_API_KEY")
    except Exception as e:
        print("OpenAI import failed; will fall back to regex heuristics.", e)
        USE_OPENAI = False


CONVERSION_FACTORS = {
    "kg": 1000,    # kg -> gram
    "KG": 1000,
    "ltr": 1000,   # liter -> ml
    "Ltr": 1000,
    "Liter": 1000,
    "Liters": 1000,
    # US gallon -> ml approx (rounded)
    "gallon": 3785,
    "gallons": 3785,
    "Gallon": 3785,
    "Gallons": 3785,
}

# After rescaling, we will map these keys to the target unit (target after multiplication)
CONVERSION_TARGET = {
    "kg": "gram",
    "KG": "gram",
    "ltr": "ml",
    "Ltr": "ml",
    "Liter": "ml",
    "Liters": "ml",
    "gallon": "ml",
    "gallons": "ml",
    "Gallon": "ml",
    "Gallons": "ml",
}

UNIT_MAP = {
    # --- invalid / missing ---
    "-": np.nan, "---": np.nan, "": np.nan, "None": np.nan, "NA": np.nan, "....": np.nan,
    "product_weight": "gram", "Foot": np.nan, "in": np.nan, "M": np.nan,
    "Sq Ft": np.nan, "sq ft": np.nan, "(Pack of 1)": np.nan, "7,2 oz": np.nan, "12.54": np.nan, "24": np.nan, "8": np.nan,

    # --- ounce ---
    "oz": "oz", "ounce": "oz", "ounces": "oz", "OZ": "oz", "Ounce": "oz", "Ounces": "oz",
    "Oz": "oz", "per Box": "oz", "per Carton": "oz",

    # --- fluid ounce ---
    "fl oz": "fl oz", "fl. oz.": "fl oz", "fl. oz": "fl oz", "Fl oz": "fl oz",
    "Fl Oz": "fl oz", "FL Oz": "fl oz", "FL OZ": "fl oz", "Fluid ounce": "fl oz",
    "Fluid Ounce": "fl oz", "Fluid Ounces": "fl oz", "fluid ounce": "fl oz",
    "fluid ounce(s)": "fl oz", "fluid ounces": "fl oz", "fluid_ounces": "fl oz",
    "Fl.oz": "fl oz", "Fl. OZ": "fl oz", "Fl OZ": "fl oz",

    # --- count / piece ---
    "count": "count", "Count": "count", "COUNT": "count", "ct": "count", "CT": "count",
    "Each": "count", "each": "count", "EA": "count", "ea": "count",
    "Piece": "count", "Pieces": "count", "Bag": "count", "bag": "count",
    "Box": "count", "box": "count", "Bottle": "count", "bottle": "count",
    "Pack": "count", "PACK": "count", "packs": "count", "Packs": "count",
    "Packet": "count", "Pouch": "count", "Carton": "count", "Container": "count",
    "Jar": "count", "jar": "count", "JARS": "count", "K-Cups": "count", "KIT": "count",
    "Capsules": "count", "Capsule": "count", "Per Package": "count",
    "Box/12": "count", "BOX/12": "count", "Bucket": "count", "Piece": "count",
    "Paper Cupcake Liners": "count", "Tea Bags": "count", "tea bags": "count",
    "bag": "count", "unit": "count", "units": "count", "stück": "count", "Cou": "count",
    "unità": "count", "Can": "count", "can": "count", "Tin": "count", "Ziplock bags": "count",
    "SACHET": "count", "Sugar Substitute": "count", "pac": "count", "Count / Count": "count",
    "bottles": "count",

    # --- grams ---
    "g": "gram", "gr": "gram", "gram": "gram", "grams": "gram", "Grams": "gram",
    "Gram": "gram", "Grams(gm)": "gram", "gramm": "gram",

    # --- milliliter ---
    "ml": "ml", "mililitro": "ml", "millilitre": "ml", "milliliter": "ml",
    "millilitro": "ml", "Milliliters": "ml", "Liter": "ml", "Liters": "ml", "Ltr": "ml", "ltr": "ml", "liters": "ml",

    # --- pounds ---
    "lb": "pound", "LB": "pound", "Lbs": "pound", "lbs": "pound",
    "Pound": "pound", "Pounds": "pound", "pounds": "pound", "pound": "pound",
}

# Allowed final normalized classes
ALLOWED_UNITS = {"oz", "fl oz", "gram", "ml", "pound", "count"}


def safe_float(s: str) -> Optional[float]:
    """Convert string to float robustly (handles commas as thousand/decimal separators)."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    # Remove surrounding non-numeric characters except ., and comma and minus
    s = s.replace("\u200b", "")  # zero width
    # Replace comma decimal like "7,2" -> "7.2" but not "1,234"
    if re.match(r"^\d+,\d+$", s):
        s = s.replace(",", ".")
    # Remove thousands separators like 1,234 or 1 234
    s = re.sub(r"[ _](?=\d{3}\b)", "", s)
    # Remove any trailing units: "16.9 oz" -> "16.9"
    m = re.search(r"([-+]?\d*\.?\d+([eE][-+]?\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def extract_value_unit_from_catalog(catalog_text: str) -> Tuple[Optional[float], Optional[str], str]:
    """
    Returns (value, unit, desc_without_last_2_lines)
    If value or unit missing, returns None for them.
    """
    if not isinstance(catalog_text, str):
        return None, None, ""
    lines = [ln.strip() for ln in catalog_text.strip().splitlines() if ln.strip() != ""]
    unit = None
    value = None
    # take last two lines if present
    if len(lines) >= 1:
        last = lines[-1]
        if last.lower().startswith("unit:"):
            unit = last.split(":", 1)[1].strip()
        else:
            # maybe unit is last line without prefix
            # sometimes the unit appears like "16.9 oz" in the unit field; handle later
            unit = last.strip()
    if len(lines) >= 2:
        second_last = lines[-2]
        if second_last.lower().startswith("value:"):
            value = safe_float(second_last.split(":", 1)[1].strip())
        else:
            # try to extract numeric from second last
            value = safe_float(second_last)
    # Build description to send to LLM (remove last two lines)
    desc_lines = lines[:-2] if len(lines) > 2 else []
    desc = "\n".join(desc_lines).strip()
    # If value is still None, try to extract numeric from last line (e.g., "1.76 Ounce" or "16.9 oz")
    if value is None and unit is not None:
        # If last line contains both number and unit
        value = safe_float(unit)
        if value is not None:
            # remove the numeric part from unit string
            # e.g. "1.76 Ounce" -> unit = "Ounce"
            unit = re.sub(r"[-+]?\d*[,\.]?\d+([eE][-+]?\d+)?", "", unit).strip()
            if unit == "":
                unit = None
    # If still missing, try to find any number in the whole catalog (first occurrence)
    if value is None:
        v = safe_float(catalog_text)
        value = v
    return value, unit, desc


def apply_conversion_if_needed(raw_unit: Optional[str], value: Optional[float]) -> Tuple[Optional[float], Optional[str]]:
    """If raw_unit is convertible (kg, ltr, gallon), rescale value and return new unit string (target)."""
    if raw_unit is None:
        return value, raw_unit
    k = raw_unit.strip()
    # exact matches in CONVERSION_FACTORS first
    if k in CONVERSION_FACTORS and value is not None:
        factor = CONVERSION_FACTORS[k]
        new_value = value * factor
        target = CONVERSION_TARGET.get(k, k)
        return new_value, target
    # try case-insensitive key
    kl = k.lower()
    for conv_key in CONVERSION_FACTORS:
        if conv_key.lower() == kl and value is not None:
            factor = CONVERSION_FACTORS[conv_key]
            new_value = value * factor
            target = CONVERSION_TARGET.get(conv_key, conv_key)
            return new_value, target
    return value, raw_unit


def lookup_unit_in_map(raw_unit: Optional[str]) -> Optional[str]:
    """
    Look up the normalized unit using UNIT_MAP.
    Attempts exact match, case-insensitive, and token-based heuristics.
    Returns one of ALLOWED_UNITS or np.nan (which is used to indicate unknown).
    """
    if raw_unit is None:
        return np.nan
    ru = raw_unit.strip()
    if ru == "":
        return np.nan
    # exact match
    if ru in UNIT_MAP:
        return UNIT_MAP[ru]
    # case-insensitive match
    for k, v in UNIT_MAP.items():
        if k.lower() == ru.lower():
            return v
    # token-based search (e.g., "16.9 oz" or "fluid ounces")
    lru = ru.lower()
    # common tokens mapping
    token_matches = {
        "fl oz": ["fl oz", "fl. oz", "fluid ounce", "fluid ounces", "fluid"],
        "oz": [" oz", "ounce", "ounces", "oz"],
        "gram": ["g ", " g", "gram", "grams", "gr", "gramm"],
        "ml": ["ml", "milliliter", "millilitre", "millilitro", "liter", "liters", "ltr"],
        "pound": ["lb", "pound", "pounds", "lbs", "l b"],
        "count": ["pack", "box", "bottle", "bag", "each", "ea", "ct", "count", "piece", "pieces", "packet", "pouch", "jar", "k-cup", "k-cups", "capsule", "capsules", "tin", "sachet"]
    }
    for target, tokens in token_matches.items():
        for t in tokens:
            if t in lru:
                return target if target in ALLOWED_UNITS else (UNIT_MAP.get(t, np.nan) if t in UNIT_MAP else target)
    return np.nan


def infer_unit_and_value_with_llm(description: str) -> Tuple[float, str]:
    """
    Try to infer value and unit from the description using a lightweight LLM (OpenAI if configured)
    or fallback to regex heuristics.
    Returns (value, unit) where unit is one of allowed units or 'count'.
    """
    desc = (description or "").strip()
    # Heuristic first — cheap and fast:
    heur_value, heur_unit = heuristic_infer(desc)
    if heur_unit is not None:
        return heur_value, heur_unit

    # If OpenAI available, try a very small request
    if USE_OPENAI:
        prompt = (
            "You are a terse helper. Given the product description below, infer the most likely numeric "
            "value and a single unit from this set: [oz, fl oz, gram, ml, pound, count].\n"
            "Return output as JSON exactly like: {\"value\": <number>, \"unit\": \"<one-of-above>\"}\n\n"
            f"Description:\n{desc}\n\nIf no value/unit found, reply {json.dumps({'value': 1, 'unit': 'count'})}."
        )
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini" if "gpt-4o-mini" in openai.Model.list().data else "gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.0,
            )
            txt = resp["choices"][0]["message"]["content"].strip()
            # parse JSON object out
            m = re.search(r"\{.*\}", txt, re.S)
            if m:
                obj = json.loads(m.group(0))
                v = safe_float(obj.get("value"))
                u = obj.get("unit")
                if u is None:
                    u = "count"
                if v is None:
                    v = 1.0
                # ensure unit is in allowed set
                u = u.strip().lower()
                # Normalize some synonyms
                if u in ("grams", "g"):
                    u = "gram"
                if u in ("ml", "milliliter", "millilitre"):
                    u = "ml"
                if u not in ALLOWED_UNITS:
                    u = "count"
                return float(v), u
        except Exception as e:
            # Fall back to heuristics
            # print("LLM inference failed:", e)
            pass

    # final fallback
    return 1.0, "count"


def heuristic_infer(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Regex-based heuristics to find patterns like '500 ml', '16 oz', '12 count', 'pack of 2', etc.
    Returns (value, unit) or (None, None) if no confident inference.
    """
    if not text:
        return None, None
    txt = text.replace("\n", " ").lower()
    # common patterns: number + optional space + unit token
    patterns = [
        (r"(\d+[\,\.]?\d*)\s*(ml|millilit(e|er)|millilitro)\b", "ml"),
        (r"(\d+[\,\.]?\d*)\s*(g|gram|grams|gr)\b", "gram"),
        (r"(\d+[\,\.]?\d*)\s*(kg|kilogram|kilograms)\b", "gram"),
        (r"(\d+[\,\.]?\d*)\s*(l|ltr|liter|liters)\b", "ml"),
        (r"(\d+[\,\.]?\d*)\s*(oz|ounce|ounces)\b", "oz"),
        (r"(\d+[\,\.]?\d*)\s*(fl\.?\s?oz|fluid ounce|fluid ounces)\b", "fl oz"),
        (r"(\d+[\,\.]?\d*)\s*(lb|pound|pounds|lbs)\b", "pound"),
        (r"(pack of|pack)\s*(\d+)", "count"),
        (r"(\d+)\s*(count|ct|pcs|pieces|pieces)\b", "count"),
        (r"(\d+)\s*(bottle|bottles|box|boxes|bag|bags|pouch|jar|jars)\b", "count"),
    ]
    for pat, unit in patterns:
        for m in re.finditer(pat, txt):
            if m:
                # number is usually in group 1 (or 2 for some)
                num = None
                # find the first numeric capture group
                for g in m.groups():
                    if isinstance(g, str) and re.search(r"\d", g):
                        num = safe_float(g)
                        if num is not None:
                            break
                if num is None:
                    # maybe group 1
                    num = safe_float(m.group(1))
                if num is not None:
                    # convert kilos/liters if unit matches kg/l/ltr via mapping below
                    if unit == "gram" and re.search(r"\bkg\b", m.group(0)):
                        num = num * 1000
                    if unit == "ml" and re.search(r"\bl\b|\bltr\b|\bliter\b", m.group(0)):
                        num = num * 1000
                    return float(num), unit
    # no match
    return None, None


def process_dataframe(df: pd.DataFrame, name: str = "df") -> pd.DataFrame:
    """
    Adds columns: raw_value, raw_unit, desc_for_llm, value_rescaled, unit_after_conversion, unit_normalized
    Then applies the >10000 rule and LLM inference for unknown units.
    """
    rows = []
    for idx, row in df.iterrows():
        catalog = row.get("catalog_content", "")
        raw_value, raw_unit, desc = extract_value_unit_from_catalog(catalog)
        # If the raw_unit is something like "16.9 oz" and we parsed value from it, raw_unit may be None; try to re-extract
        # Apply conversion factors if needed
        value_after_conv, unit_after_conv = apply_conversion_if_needed(raw_unit, raw_value)
        rows.append({
            "_idx": idx,
            "raw_value": raw_value,
            "raw_unit": raw_unit,
            "desc_for_llm": desc,
            "value_after_conv": value_after_conv,
            "unit_after_conv": unit_after_conv
        })
    meta = pd.DataFrame(rows).set_index("_idx")

    # Merge meta back into df (safe)
    out = df.copy()
    out = out.join(meta, how="left")

    # First pass: mark arbitrary big values (>10000) -> set to NaN and unit to None
    big_mask = out["value_after_conv"].apply(lambda x: (x is not None) and (not (isinstance(x, float) or isinstance(x, int)) and not np.isnan(x)) == False and (x is not None) and (not math.isnan(x)) and (float(x) > 10000) if x is not None else False)
    # The above lambda is slightly defensive; simpler:
    big_mask = out["value_after_conv"].apply(lambda x: isinstance(x, (int, float)) and (not math.isnan(x)) and (float(x) > 10000))
    n_big = big_mask.sum()
    print(f"[{name}] First pass: {int(n_big)} rows have value > 10000 and will be set to NaN (value) + None (unit).")
    out.loc[big_mask, "value_after_conv"] = np.nan
    out.loc[big_mask, "unit_after_conv"] = None
    out.loc[big_mask, "unit_normalized"] = np.nan

    # Map remaining units using UNIT_MAP / lookup
    def normalize_row_unit(u):
        mapped = lookup_unit_in_map(u)
        # If lookup hits a key that maps to np.nan, keep np.nan
        # If mapped is one of our allowed labels, return it
        if isinstance(mapped, str) and mapped in ALLOWED_UNITS:
            return mapped
        # Some UNIT_MAP entries use string keys mapping directly to final (e.g., "g" -> "gram")
        if isinstance(mapped, str) and mapped in {"gram", "ml", "pound", "count", "oz", "fl oz"}:
            # unify 'gram' -> 'gram' (we keep as-is)
            # ensure 'fl oz' spelled correctly
            if mapped == "gram":
                return "gram"
            if mapped == "ml":
                return "ml"
            if mapped == "pound":
                return "pound"
            if mapped == "count":
                return "count"
            if mapped == "oz":
                return "oz"
            if mapped == "fl oz":
                return "fl oz"
        return np.nan

    out["unit_normalized"] = out["unit_after_conv"].apply(normalize_row_unit)

    # Now for rows where unit_normalized is np.nan, invoke LLM or heuristic
    need_infer_mask = out["unit_normalized"].isna()
    n_need_infer = int(need_infer_mask.sum())
    print(f"[{name}] {n_need_infer} rows need LLM/heuristic inference for unit/value (unit normalized is NaN).")

    inferred_values = []
    inferred_units = []
    for idx, row in out[need_infer_mask].iterrows():
        desc = row.get("desc_for_llm", "") or ""
        v_inf, u_inf = infer_unit_and_value_with_llm(desc)
        # ensure unit normalized is one of allowed
        if u_inf not in ALLOWED_UNITS:
            u_inf = "count"
        inferred_values.append((idx, float(v_inf)))
        inferred_units.append((idx, u_inf))

    # Apply inferred values/units back
    for idx, v in inferred_values:
        out.at[idx, "value_after_conv"] = v
    for idx, u in inferred_units:
        out.at[idx, "unit_after_conv"] = u
        out.at[idx, "unit_normalized"] = u

    # Final check: ensure unit_normalized in allowed set or NaN
    out["unit_normalized"] = out["unit_normalized"].apply(lambda u: u if (isinstance(u, str) and u in ALLOWED_UNITS) else (np.nan if pd.isna(u) else (u if u in ALLOWED_UNITS else np.nan)))

    # Summarize counts by normalized unit
    summary = out["unit_normalized"].value_counts(dropna=False)
    print(f"[{name}] Normalized unit distribution:\n{summary.to_string()}\n")
    return out


def main():
    files = ["/kaggle/input/amazon-ml-dataset-csv/splits/splits/train.csv", "/kaggle/input/amazon-ml-dataset-csv/splits/splits/val.csv", "/kaggle/input/amazon-ml-dataset-csv/dataset/dataset/test.csv"]
    output_files = ["train_split_norm.csv", "val_split_norm.csv", "test_norm.csv"]
    for i, f in enumerate(files):
        if not os.path.exists(f):
            print(f"File {f} not found in current directory; skipping.")
            continue
        print(f"\nProcessing {f} ...")
        df = pd.read_csv(f)
        processed = process_dataframe(df, name=f)
        # Save processed CSV for inspection
        out_name = output_files[i]
        processed.to_csv(out_name, index=False)
        print(f"Saved normalized output to {out_name}")


if __name__ == "__main__":
    main()
