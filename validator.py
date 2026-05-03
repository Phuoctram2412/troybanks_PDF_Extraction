# If a new field needs to be added, update ALL_FIELDS and AMOUNT_FIELDS below

# If a new field needs to be added, update ALL_FIELDS and AMOUNT_FIELDS below

import re


def validate_and_clean(raw: dict) -> tuple[dict, dict]:
    """
    Cleans extracted values and builds confidence level labels.
    Field names match the bills table column names exactly.
    """
    # Financial fields stored as REAL in SQLite — Gemini should already
    # return these as numbers, but strip $ and commas as a safety net
    AMOUNT_FIELDS = {"amount_due", "usage_quantity"}

    NULL_STRINGS = {"n/a", "none", "null", "—", "-", "na", "not shown", ""}

    confidence_scores = raw.pop("confidence_scores", {})
    cleaned = {}

    for key, value in raw.items():
        # Normalise null-like strings to None
        if isinstance(value, str) and value.strip().lower() in NULL_STRINGS:
            value = None

        # Strip $ and commas from amount fields — safety net in case
        # Gemini includes them despite the prompt saying not to
        if value is not None and key in AMOUNT_FIELDS:
            if isinstance(value, str):
                cleaned_val = value.replace("$", "").replace(",", "").strip()
                try:
                    value = float(cleaned_val)
                except ValueError:
                    value = None   # reject garbled amounts

        cleaned[key] = value

    # Build confidence level labels
    # Keep this list in sync with BILL_SCHEMA in gemini_client.py
    # Build confidence level labels
# Keep this list in sync with BILL_SCHEMA in gemini_client.py
    ALL_FIELDS = [
        "provider_name",
        "customer_name",
        "account_number",
        "bill_date",
        "due_date",
        "meter_number",
        "usage_quantity",
        "usage_unit",
        "amount_due",
    ]

    confidence_levels = {}
    for field in ALL_FIELDS:
        score = confidence_scores.get(field)
        if score is None:
            level = "UNKNOWN"
        elif score >= 0.85:
            level = "HIGH"
        elif score >= 0.60:
            level = "MEDIUM"
        else:
            level = "LOW"
        confidence_levels[field] = {"score": score, "level": level}

    return cleaned, confidence_levels