# # If Gemini change their API, modify this code

import json
import os
import time
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL          = "gemini-2.5-flash"
MAX_RETRIES    = 4


# ─────────────────────────────────────────────────────────────────────────────
# Type helpers
# ─────────────────────────────────────────────────────────────────────────────
# Two helpers — one per DB column type.
# STRING  → maps to TEXT   columns in SQLite
# NUMBER  → maps to REAL   columns in SQLite
#
# Every field is nullable=True because any field can be absent on any bill.
# Gemini returns null for missing fields rather than guessing.

def _string(description: str) -> dict:
    """Maps to TEXT column in SQLite."""
    return {
        "type":        "STRING",
        "nullable":    True,
        "description": description
    }

def _number(description: str) -> dict:
    """Maps to REAL column in SQLite. Gemini returns a bare float, no $ sign."""
    return {
        "type":        "NUMBER",
        "nullable":    True,
        "description": description
    }


# ─────────────────────────────────────────────────────────────────────────────
# BILL_SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
# Field names here match the bills table column names exactly.
# Types here match the bills table column types exactly.
# This means db_handler.py can do a direct one-to-one mapping with no
# type conversion — what Gemini returns goes straight into the database.
#
# Column type reference (from create_db.py):
#   TEXT  → customer_name, account_number, service_address
#           bill_date, due_date, service_period_start, service_period_end
#           meter_number, previous_read_date, previous_reading
#           current_read_date, present_reading, usage_unit
#   REAL  → usage_quantity
#           previous_balance, payments_applied, current_charges
#           taxes_and_fees, adjustments, amount_due

BILL_SCHEMA = {
    "type": "OBJECT",
    "properties": {

        # ── Identity  (TEXT → STRING) ──────────────────────────────────────
        "provider_name": _string(
            "Full legal name of the utility company exactly as printed on the "
            "bill letterhead e.g. Washington Water Service, Con Edison, "
            "National Grid, We Energies. Read it directly from the header or "
            "logo area of the bill."
        ),
        "customer_name": _string(
            "Full name of the account holder exactly as shown on the bill"
        ),
        "account_number": _string(
            "Account or customer number — preserve exact format including "
            "hyphens e.g. 67-7512-2755-0004-5 or 0000000000"
        ),

        # ── Timeline  (TEXT → STRING) ──────────────────────────────────────
        "bill_date": _string(
            "Date the bill was issued — YYYY-MM-DD format e.g. 2021-04-13"
        ),
        "due_date": _string(
            "Payment due date — YYYY-MM-DD format e.g. 2021-07-19. "
            "May be labelled Pay By, Due Date, or Please Pay By"
        ),

        # ── Consumption  (TEXT/NUMBER) ─────────────────────────────────────
        "meter_number": _string(
            "Meter or device identifier exactly as shown — numeric or "
            "alphanumeric"
        ),
        "usage_quantity": _number(
            "Total usage this billing period — numeric only, no units. "
            "e.g. 1414 for 1414 CF,  228 for 228 kWh,  14.5 for 14.5 therms"
        ),
        "usage_unit": _string(
            "Unit of measurement for usage_quantity. "
            "Use exactly one of: CF  CCF  kWh  therms  gallons"
        ),

        # ── Financials  (REAL → NUMBER) ────────────────────────────────────
        "amount_due": _number(
            "Total amount due or please pay amount — numeric only e.g. 75.64. "
            "This is the final amount the customer must pay"
        ),

        # ── Confidence scores ──────────────────────────────────────────────
        # One score per extracted field — 0.0 to 1.0
        # 1.0 = field clearly visible, no ambiguity
        # 0.7 = found but slightly uncertain (poor print quality, etc.)
        # 0.4 = inferred from context, not directly stated
        # 0.0 = field not found on this bill
        "confidence_scores": {
            "type":        "OBJECT",
            "description": "Confidence score 0.0-1.0 for each extracted field",
            "properties": {
                field: {"type": "NUMBER", "nullable": True}
                for field in [
                    "provider_name",      # ← FIXED: comma was missing
                    "customer_name",
                    "account_number",
                    "bill_date",
                    "due_date",
                    "meter_number",
                    "usage_quantity",
                    "usage_unit",
                    "amount_due",
                ]
            }
        }
    },

    # Every field must appear in the response — missing fields return null
    "required": [
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
}


# Prompt updated to match the 9-field schema — removed references to
# financial breakdown fields (previous_balance, payments_applied, etc.)
# and date-range fields (service_period_*, previous_read_date, etc.) that
# aren't in the schema. Gemini was being told to extract fields it had no
# place to put, leading to wasted tokens and confusion.
EXTRACTION_PROMPT = """You are a billing data extraction specialist for TroyBanks audit operations.

Extract every billing field from this utility bill image and return structured JSON.

Rules:
- provider_name: read the utility company name exactly as printed on the
  bill letterhead or logo. Do not abbreviate or guess — use the exact
  name shown.
- amount_due: return as a bare number only — no $ sign, no commas
  e.g. 75.64 not $75.64, 1234.56 not $1,234.56
- bill_date and due_date: use YYYY-MM-DD format e.g. 2021-04-13.
  Convert whatever date format you see on the bill into this format.
- account_number: preserve the exact format shown including hyphens
- usage_quantity is the number only e.g. 1414 — the unit goes in usage_unit
- usage_unit must be exactly one of: CF  CCF  kWh  therms  gallons
- meter_number: preserve exact format including any hyphens or letters
- If a field is genuinely not present on this bill return null — do not guess
- confidence_scores: rate 0.0-1.0 how clearly each field appeared in the image"""


# ─────────────────────────────────────────────────────────────────────────────
# API call
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(page_bytes_list: list[bytes],
                model: str = MODEL,
                retries: int = MAX_RETRIES) -> dict:
    client     = genai.Client(api_key=GEMINI_API_KEY)
    parts      = []
    last_error = None

    for i, page_bytes in enumerate(page_bytes_list):
        parts.append(types.Part.from_bytes(data=page_bytes, mime_type="image/png"))
        if len(page_bytes_list) > 1:
            parts.append(types.Part.from_text(text=f"[Page {i+1} of {len(page_bytes_list)}]"))
    parts.append(types.Part.from_text(text=EXTRACTION_PROMPT))

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BILL_SCHEMA,
                    temperature=0.0,
                    max_output_tokens=8192,
                )
            )
            return json.loads(response.text)

        except Exception as e:
            last_error  = e
            error_str   = str(e)

            if "per_day" in error_str or "free_tier" in error_str.lower():
                raise RuntimeError(
                    "Daily free tier quota exhausted (20 req/day for gemini-2.5-flash). "
                    "Wait until tomorrow, switch to gemini-2.5-flash-lite, "
                    "or add billing at aistudio.google.com"
                )
            elif "429" in error_str:
                wait = 60
                print(f"  Rate limit hit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                if attempt < retries:
                    wait = 2 ** attempt
                    print(f"  Attempt {attempt} failed. Retrying in {wait}s...")
                    time.sleep(wait)

    raise RuntimeError(f"Gemini API failed after {retries} attempts: {last_error}")