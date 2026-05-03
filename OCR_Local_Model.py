# """
# app1.py — Auditor-facing UI for utility bill field extraction.

# Run with:
#     streamlit run app1.py

# Requirements:
#     pip install streamlit pandas requests pillow pymupdf
#     Ollama running locally on :11434 with llama3.1 pulled.
#     test_pdf_function.process_pdf available on the import path.
# """

# import io
# import json
# import tempfile
# from pathlib import Path

# import pandas as pd
# import requests
# import streamlit as st
# from PIL import Image
# import re
# from test_pdf_function import process_pdf

# # Optional PDF rendering — if not installed, PDF previews fall back to a download button.
# try:
#     import fitz  # PyMuPDF
#     HAS_PYMUPDF = True
# except ImportError:
#     HAS_PYMUPDF = False


# # ---------------------------------------------------------------------------
# # Configuration
# # ---------------------------------------------------------------------------

# OLLAMA_URL = "http://localhost:11434/api/chat"
# # MODEL_NAME = "qwen2.5vl:3b"
# MODEL_NAME = "llama3.1:latest"
# REQUEST_TIMEOUT = 120  # seconds


# # ---------------------------------------------------------------------------
# # Bill type detection + unit constraints
# # ---------------------------------------------------------------------------
# # Keyword scoring is more robust than a single keyword check — one word
# # might appear coincidentally on any bill, but several matching keywords
# # is strong evidence. The bill type then constrains usage_unit so that
# # the model literally cannot return "kWh" on a water bill.

# UNITS_BY_BILL_TYPE = {
#     "water":         ["CF", "CCF", "gallons"],
#     "electric":      ["kWh"],
#     "gas":           ["therms", "CCF"],   # gas can be billed in either
#     "national_grid": ["kWh", "therms"],   # dual-service bill
#     "unknown":       ["CF", "CCF", "kWh", "therms", "gallons"],  # all options
# }

# BILL_TYPE_DESCRIPTIONS = {
#     "water":         "WATER utility bill (units will be CF, CCF, or gallons)",
#     "electric":      "ELECTRIC utility bill (units will be kWh)",
#     "gas":           "NATURAL GAS utility bill (units will be therms or CCF)",
#     "national_grid": "National Grid bill — electric and/or gas service "
#                      "(units will be kWh or therms)",
#     "unknown":       "utility bill (type could not be determined from header)",
# }

# def clean_bill_text(raw_text: str) -> str:
#     """
#     Cleans raw OCR text before sending to the model. The goal is to
#     reduce noise and token count without losing any field values.

#     Five passes, each cheap:
#       1. Strip control characters that confuse models (\f, \x0b, etc)
#       2. Strip trailing whitespace on every line
#       3. Drop lines that are pure noise (single stray characters)
#       4. Collapse runs of 3+ blank lines down to 1
#       5. Collapse runs of 3+ spaces within a line down to 2 spaces
#          (we keep 2 spaces because column alignment occasionally matters
#          for OCR layout reasoning — e.g. "Pay By  07/19/21")

#     Returns cleaned text. On a typical bill this trims 20-40% of the
#     character count and produces fewer "wasted" tokens for the model
#     to read past.
#     """
#     if not raw_text:
#         return ""

#     # Pass 1 — strip form feeds and other control characters
#     # Form feeds appear between PDF pages; vertical tabs from some scanners
#     text = raw_text.replace("\f", "\n").replace("\x0b", "\n").replace("\x0c", "\n")
#     # Pass 2 — strip trailing whitespace on every line
#     # Without this, lines often end with 5-10 spaces that consume tokens
#     lines = [line.rstrip() for line in text.split("\n")]

#     # Pass 3 — drop lines that are pure noise
#     # A line of just "|", "~", "...", "***" etc is OCR border artefact —
#     # remove it entirely. Keep any line that has at least one alphanumeric
#     # character so we never accidentally drop a value.
#     NOISE_PATTERN = re.compile(r"^[\s\W_]*$")  # only non-word chars or empty
#     cleaned_lines = []
#     for line in lines:
#         # Keep blank lines (handled in pass 4) and lines with real content
#         if line == "" or not NOISE_PATTERN.match(line):
#             cleaned_lines.append(line)
#         # else: pure-noise line dropped silently

#     text = "\n".join(cleaned_lines)

#     # Pass 4 — collapse runs of 3+ blank lines down to 1 blank line
#     # Two consecutive newlines = paragraph break (kept). More than that
#     # was just OCR whitespace — collapse to keep token count down.
#     text = re.sub(r"\n{3,}", "\n\n", text)

#     # Pass 5 — collapse 3+ spaces to 2 within a line
#     # Keep at least 2 spaces because some OCR layout reasoning depends on
#     # column alignment (e.g. "Pay By   07/19/21" — the gap signals column).
#     # But 5+ spaces is wasteful — bring it down to 2.
#     text = re.sub(r" {3,}", "  ", text)

#     return text.strip()

# def detect_bill_type(text: str) -> str:
#     """
#     Scans OCR text for distinctive keywords identifying the utility type.
#     Returns "water", "electric", "gas", "national_grid", or "unknown".
#     """
#     text_lower = text.lower()

#     scores = {
#         "water": sum(1 for kw in
#             ["water service", "cubic feet", " cf ", "water metered",
#              "washington water", "gallons", "water department"]
#             if kw in text_lower),
#         "electric": sum(1 for kw in
#             ["kwh", "kilowatt", "electric", "con edison", "coned",
#              "electric supply", "electricity charges"]
#             if kw in text_lower),
#         "gas": sum(1 for kw in
#             ["therms", "natural gas", "gas service", "gas delivery"]
#             if kw in text_lower and "national grid" not in text_lower),
#     }

#     detected  = max(scores, key=scores.get)
#     top_score = scores[detected]

#     if top_score == 0:
#         return "unknown"
#     return detected


# # ---------------------------------------------------------------------------
# # Schema and prompt — built dynamically based on detected bill type
# # ---------------------------------------------------------------------------

# SYSTEM_PROMPT = """You are a strict data extraction tool. You do not interpret, infer, summarize, or guess.

# ABSOLUTE RULES:
# 1. Extract ONLY values that appear VERBATIM in the source text. If a value is not literally present in the text, return null for that field. Do not fabricate.
# 2. Do NOT use outside knowledge. Do NOT fill in plausible-looking values. Do NOT correct apparent typos.
# 3. Do NOT combine, calculate, or derive values. If the text says "Total: $100" do not infer due amount from line items — copy the stated total.
# 4. If a field appears multiple times with conflicting values, return null for that field.
# 5. If you are not 100% certain a value in the text corresponds to the requested field, return null. Uncertainty = null.
# 6. Return ONLY the JSON object. No prose, no explanations, no markdown fences, no commentary."""


# def build_schema(bill_type: str) -> dict:
#     """
#     Builds the JSON schema with usage_unit constrained to valid units for
#     this bill type. Ollama enforces enum constraints at the structured-output
#     level — the model literally cannot return an invalid unit value.
#     """
#     valid_units = UNITS_BY_BILL_TYPE.get(bill_type, UNITS_BY_BILL_TYPE["unknown"])

#     return {
#         "type": "object",
#         "properties": {
#             "provider_name":  {"type": ["string", "null"]},
#             "customer_name":  {"type": ["string", "null"]},
#             "account_number": {"type": ["string", "null"]},
#             "bill_date":      {"type": ["string", "null"]},
#             "due_date":       {"type": ["string", "null"]},
#             "amount_due":     {"type": ["number", "null"]},
#             "usage_quantity": {"type": ["number", "null"]},
#             "usage_unit": {
#                 # enum constraint — model is restricted to these values + null
#                 "type": ["string", "null"],
#                 "enum": valid_units + [None],
#             },
#             "meter_number":   {"type": ["string", "null"]},
#         },
#         "required": [
#             "provider_name", "customer_name", "account_number",
#             "bill_date", "due_date", "amount_due",
#             "usage_quantity", "usage_unit", "meter_number",
#         ],
#         "additionalProperties": False,
#     }


# def build_user_prompt(bill_text: str, bill_type: str) -> str:
#     """
#     Leaner version — the schema enum already constrains usage_unit at the
#     output level so the prompt only needs a one-line bill type hint, not
#     extensive per-unit examples. This trims ~600 tokens of prompt overhead.
#     """
#     valid_units = UNITS_BY_BILL_TYPE.get(bill_type, UNITS_BY_BILL_TYPE["unknown"])
#     units_str   = ", ".join(valid_units)

#     return f"""Extract these fields from the utility bill text below.

# This is a {bill_type} bill — usage_unit must be one of: {units_str}.

# FIELDS:

# - provider_name (string | null): utility company name from header,
#   exactly as printed e.g. "Con Edison", "National Grid".
# - customer_name (string | null): account holder's name as written.
# - account_number (string | null): preserve exact format including hyphens.
# - bill_date (string | null): date bill was issued, YYYY-MM-DD format.
# - due_date (string | null): payment due date, YYYY-MM-DD. Labels: "PAY BY",
#   "DUE DATE", "Please Pay By". If label and value are on different lines
#   due to OCR layout, look 1-2 lines below the label for the date.
# - amount_due (number | null)
#   Match: the total amount the customer owes for this bill, as a bare number, no $ sign.
#   Source labels (in order of preference):
#     "TOTAL AMOUNT DUE", "Total amount due", "Amount Due", "AMOUNT DUE",
#     "Balance Due", "Please Pay", "Current balance due", "Total Due",
#     "Pay This Amount", "PAY THIS AMOUNT"
# - usage_quantity (number | null)
#   Look for value that near the "Total Usage" text
#   Another case is the amount is near the Usage Unit.
#   If you know the type of bill then use the Unit as the reference to find the number around that. That is your unit usage.
#   The result is right above the text "Total Electric Usage".

# - usage_unit (string | null): one of {units_str} for this {bill_type} bill.
# - meter_number (string | null): meter identifier as written.
#   Do NOT confuse with meter readings.

# Reformat any literal date to YYYY-MM-DD. Do NOT invent values.
# Return ONLY the JSON object.

# BILL TEXT:
# \"\"\"
# {bill_text}
# \"\"\"

# Return the JSON object now."""


# # ---------------------------------------------------------------------------
# # Core extraction
# # ---------------------------------------------------------------------------

# def extract_bill_text(file_path: str) -> str:
#     """
#     Run the OCR pipeline and return the cleaned text.
#     Cleaning removes whitespace noise and OCR border artefacts so the
#     model can focus on actual content.
#     """
#     result    = process_pdf(file_path)
#     raw_text  = result["pages"][-1]["full_text"]
#     return clean_bill_text(raw_text)


# def extract_fields(bill_text: str) -> tuple[dict, str]:
#     """
#     Detects bill type from OCR text, builds the type-constrained schema
#     and prompt, then sends to Ollama.

#     Returns (extracted_fields_dict, detected_bill_type).
#     The bill type is returned alongside so the UI can show what the
#     detection picked up.
#     """
#     # Detect bill type FIRST — used to constrain unit values
#     bill_type = detect_bill_type(bill_text)

#     # Build schema and prompt with the unit constraint baked in
#     schema = build_schema(bill_type)
#     prompt = build_user_prompt(bill_text, bill_type)

#     response = requests.post(
#         OLLAMA_URL,
#         json={
#             "model": MODEL_NAME,
#             "messages": [
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user",   "content": prompt},
#             ],
#             "format": schema,    # enum-constrained — model can't return invalid unit
#             "stream": False,
#             "options": {
#                 "temperature": 0,
#                 "top_p": 0.1,
#                 "num_ctx": 8192,
#                 "seed": 1704,
#             },
#         },
#         timeout=REQUEST_TIMEOUT,
#     )
#     response.raise_for_status()
#     return json.loads(response.json()["message"]["content"]), bill_type


# def process_one_file(uploaded_file) -> dict:
#     """Save upload to a temp path, extract text, run the model. Returns a row dict."""
#     suffix = Path(uploaded_file.name).suffix or ".pdf"
#     file_bytes = uploaded_file.getvalue()  # keep raw bytes for the side-by-side preview

#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         tmp.write(file_bytes)
#         tmp_path = tmp.name

#     try:
#         bill_text         = extract_bill_text(tmp_path)

#         fields, bill_type = extract_fields(bill_text)
#         return {
#             "source_file":    uploaded_file.name,
#             "bill_type":      bill_type,
#             "provider_name":  fields.get("provider_name"),
#             "customer_name":  fields.get("customer_name"),
#             "bill_date":      fields.get("bill_date"),
#             "due_date":       fields.get("due_date"),
#             "amount_due":     fields.get("amount_due"),
#             "account_number": fields.get("account_number"),
#             "meter_number":   fields.get("meter_number"),
#             "usage_quantity": fields.get("usage_quantity"),
#             "usage_unit":     fields.get("usage_unit"),
#             "status":         "OK",
#             "error":          None,
#             "_bill_text":     bill_text,
#             "_file_bytes":    file_bytes,
#         }
#     except Exception as e:
#         return {
#             "source_file":    uploaded_file.name,
#             "bill_type":      None,
#             "provider_name":  None,
#             "customer_name":  None,
#             "bill_date":      None,
#             "due_date":       None,
#             "amount_due":     None,
#             "account_number": None,
#             "meter_number":   None,
#             "usage_quantity": None,
#             "usage_unit":     None,
#             "status":         "ERROR",
#             "error":          str(e),
#             "_bill_text":     "",
#             "_file_bytes":    file_bytes,
#         }
#     finally:
#         Path(tmp_path).unlink(missing_ok=True)


# def render_original(file_bytes: bytes, file_name: str, max_pages: int = 5):
#     """Return a list of PIL images for displaying the original document."""
#     suffix = Path(file_name).suffix.lower()

#     if suffix in {".png", ".jpg", ".jpeg"}:
#         return [Image.open(io.BytesIO(file_bytes))]

#     if suffix == ".pdf":
#         if not HAS_PYMUPDF:
#             return None
#         images = []
#         with fitz.open(stream=file_bytes, filetype="pdf") as doc:
#             for page in doc[:max_pages]:
#                 pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
#                 images.append(Image.open(io.BytesIO(pix.tobytes("png"))))
#         return images

#     return []


# # ---------------------------------------------------------------------------
# # UI
# # ---------------------------------------------------------------------------

# st.set_page_config(page_title="Bill Field Extractor", page_icon="📄", layout="wide")

# st.title("📄 Bill Field Extractor")
# st.caption("Upload one or more utility bills (PDF or image). Extracted fields appear in the table below.")

# with st.sidebar:
#     st.subheader("Settings")
#     st.text_input("Ollama URL", value=OLLAMA_URL, key="ollama_url", disabled=True)
#     st.text_input("Model", value=MODEL_NAME, key="model_name", disabled=True)
#     st.markdown("---")
#     st.markdown("**Tip:** make sure `ollama serve` is running and the model is pulled.")
#     st.markdown("---")
#     st.markdown(
#         "**Bill type detection:** the OCR text is scanned for keywords "
#         "to identify whether the bill is water, electric, gas, or "
#         "National Grid. The detected type constrains which usage units "
#         "the model is allowed to return — preventing impossible answers "
#         "like `kWh` on a water bill."
#     )

# uploaded_files = st.file_uploader(
#     "Drop bills here",
#     type=["pdf", "png", "jpg", "jpeg"],
#     accept_multiple_files=True,
# )

# if "results" not in st.session_state:
#     st.session_state.results = None

# col_btn1, col_btn2, _ = st.columns([1, 1, 3])
# with col_btn1:
#     extract_clicked = st.button(
#         "🚀 Extract fields",
#         type="primary",
#         disabled=not uploaded_files,
#         use_container_width=True,
#     )
# with col_btn2:
#     clear_clicked = st.button(
#         "🗑️ Clear results",
#         disabled=st.session_state.results is None,
#         use_container_width=True,
#     )

# if clear_clicked:
#     st.session_state.results = None
#     st.rerun()

# if extract_clicked and uploaded_files:
#     rows = []
#     progress = st.progress(0.0, text="Starting...")
#     for i, f in enumerate(uploaded_files, start=1):
#         progress.progress(
#             (i - 1) / len(uploaded_files),
#             text=f"Processing {f.name} ({i}/{len(uploaded_files)})",
#         )
#         rows.append(process_one_file(f))
#     progress.progress(1.0, text="Done")
#     progress.empty()
#     st.session_state.results = rows

# if st.session_state.results:
#     rows = st.session_state.results
#     df = pd.DataFrame(rows)

#     total  = len(df)
#     ok     = (df["status"] == "OK").sum()
#     failed = total - ok
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Total bills", total)
#     col2.metric("Extracted", ok)
#     col3.metric("Failed", failed)

#     st.subheader("Extracted records")
#     visible_cols = [
#         "source_file", "bill_type", "provider_name", "customer_name",
#         "bill_date", "due_date", "amount_due", "account_number",
#         "meter_number", "usage_quantity", "usage_unit", "status",
#     ]
#     st.dataframe(df[visible_cols], use_container_width=True, hide_index=True)

#     csv_bytes = df[visible_cols].to_csv(index=False).encode("utf-8")
#     st.download_button(
#         "⬇️ Download as CSV",
#         data=csv_bytes,
#         file_name="extracted_bills.csv",
#         mime="text/csv",
#     )

#     with st.expander("🔍 Per-file details — original vs. extracted"):
#         for idx, row in enumerate(rows):
#             st.markdown(f"### {row['source_file']} — `{row['status']}`  ·  detected: `{row.get('bill_type', '?')}`")

#             if row["error"]:
#                 st.error(row["error"])

#             col_original, col_extracted = st.columns(2)

#             with col_original:
#                 st.markdown("**Original document**")
#                 images = render_original(row["_file_bytes"], row["source_file"])

#                 if images is None:
#                     st.info(
#                         "PDF preview requires PyMuPDF. Install it with "
#                         "`pip install pymupdf` to see the original here."
#                     )
#                 elif images:
#                     for page_idx, img in enumerate(images, start=1):
#                         if len(images) > 1:
#                             st.caption(f"Page {page_idx}")
#                         st.image(img, use_container_width=True)
#                 else:
#                     st.warning("Unsupported file type for preview.")

#                 st.download_button(
#                     "⬇️ Download original",
#                     data=row["_file_bytes"],
#                     file_name=row["source_file"],
#                     key=f"dl_{idx}_{row['source_file']}",
#                 )

#             with col_extracted:
#                 st.markdown("**Extracted text**")
#                 if row["_bill_text"]:
#                     st.text_area(
#                         "Extracted text",
#                         value=row["_bill_text"],
#                         height=400,
#                         key=f"text_{idx}_{row['source_file']}",
#                         label_visibility="collapsed",
#                     )
#                 else:
#                     st.caption("(no text extracted)")

#                 st.markdown("**Extracted fields**")
#                 amount = row["amount_due"]
#                 usage_str = (
#                     f"{row['usage_quantity']} {row['usage_unit']}"
#                     if row.get("usage_quantity") is not None
#                     else "—"
#                 )
#                 fields_df = pd.DataFrame({
#                     "Field": [
#                         "Bill type (detected)", "Provider", "Customer name",
#                         "Bill date", "Due date", "Amount due",
#                         "Account number", "Meter number", "Usage",
#                     ],
#                     "Value": [
#                         row.get("bill_type") or "—",
#                         row.get("provider_name") or "—",
#                         row.get("customer_name") or "—",
#                         row.get("bill_date") or "—",
#                         row.get("due_date") or "—",
#                         f"${amount:.2f}" if amount is not None else "—",
#                         row.get("account_number") or "—",
#                         row.get("meter_number") or "—",
#                         usage_str,
#                     ],
#                 })
#                 st.table(fields_df)

#             st.markdown("---")
# elif uploaded_files:
#     st.info(
#         f"📥 {len(uploaded_files)} file(s) ready. "
#         "Click **🚀 Extract fields** above to process."
#     )
# else:
#     st.info("Upload one or more bill files to begin.")



"""
app1.py — Auditor-facing UI for utility bill field extraction.

Run with:
    streamlit run app1.py

Requirements:
    pip install streamlit pandas requests pillow pymupdf
    Ollama running locally on :11434 with llama3.1 pulled.
    test_pdf_function.process_pdf available on the import path.
"""

import io
import json
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image
import re
from test_pdf_function import process_pdf

# Optional PDF rendering — if not installed, PDF previews fall back to a download button.
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Database integration — uses the same db_handler as the Gemini app
# so bills extracted by either path land in the same SQLite database
# with the same schema. Provider and customer auto-creation, duplicate
# detection by (account, bill_date), date normalisation are all handled
# inside save_bill_to_db.
from database.db_handler import save_bill_to_db


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
# MODEL_NAME = "qwen2.5vl:3b"
MODEL_NAME = "llama3.1:latest"
REQUEST_TIMEOUT = 120  # seconds
DB_PATH = "troybanks_bills.db"


# ---------------------------------------------------------------------------
# Bill type detection + unit constraints
# ---------------------------------------------------------------------------
# Keyword scoring is more robust than a single keyword check — one word
# might appear coincidentally on any bill, but several matching keywords
# is strong evidence. The bill type then constrains usage_unit so that
# the model literally cannot return "kWh" on a water bill.

UNITS_BY_BILL_TYPE = {
    "water":         ["CF", "CCF", "gallons"],
    "electric":      ["kWh"],
    "gas":           ["therms", "CCF"],   # gas can be billed in either
    "national_grid": ["kWh", "therms"],   # dual-service bill
    "unknown":       ["CF", "CCF", "kWh", "therms", "gallons"],  # all options
}

BILL_TYPE_DESCRIPTIONS = {
    "water":         "WATER utility bill (units will be CF, CCF, or gallons)",
    "electric":      "ELECTRIC utility bill (units will be kWh)",
    "gas":           "NATURAL GAS utility bill (units will be therms or CCF)",
    "national_grid": "National Grid bill — electric and/or gas service "
                     "(units will be kWh or therms)",
    "unknown":       "utility bill (type could not be determined from header)",
}

def clean_bill_text(raw_text: str) -> str:
    """
    Cleans raw OCR text before sending to the model. The goal is to
    reduce noise and token count without losing any field values.
    """
    if not raw_text:
        return ""

    text = raw_text.replace("\f", "\n").replace("\x0b", "\n").replace("\x0c", "\n")
    lines = [line.rstrip() for line in text.split("\n")]

    NOISE_PATTERN = re.compile(r"^[\s\W_]*$")
    cleaned_lines = []
    for line in lines:
        if line == "" or not NOISE_PATTERN.match(line):
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {3,}", "  ", text)

    return text.strip()

def detect_bill_type(text: str) -> str:
    """
    Scans OCR text for distinctive keywords identifying the utility type.
    Returns "water", "electric", "gas", "national_grid", or "unknown".
    """
    text_lower = text.lower()

    scores = {
        "water": sum(1 for kw in
            ["water service", "cubic feet", " cf ", "water metered",
             "washington water", "gallons", "water department"]
            if kw in text_lower),
        "electric": sum(1 for kw in
            ["kwh", "kilowatt", "electric", "con edison", "coned",
             "electric supply", "electricity charges"]
            if kw in text_lower),
        "gas": sum(1 for kw in
            ["therms", "natural gas", "gas service", "gas delivery"]
            if kw in text_lower and "national grid" not in text_lower),
    }

    detected  = max(scores, key=scores.get)
    top_score = scores[detected]

    if top_score == 0:
        return "unknown"
    return detected


# ---------------------------------------------------------------------------
# Schema and prompt — built dynamically based on detected bill type
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a strict data extraction tool. You do not interpret, infer, summarize, or guess.

ABSOLUTE RULES:
1. Extract ONLY values that appear VERBATIM in the source text. If a value is not literally present in the text, return null for that field. Do not fabricate.
2. Do NOT use outside knowledge. Do NOT fill in plausible-looking values. Do NOT correct apparent typos.
3. Do NOT combine, calculate, or derive values. If the text says "Total: $100" do not infer due amount from line items — copy the stated total.
4. If a field appears multiple times with conflicting values, return null for that field.
5. If you are not 100% certain a value in the text corresponds to the requested field, return null. Uncertainty = null.
6. Return ONLY the JSON object. No prose, no explanations, no markdown fences, no commentary."""


def build_schema(bill_type: str) -> dict:
    """
    Builds the JSON schema with usage_unit constrained to valid units for
    this bill type. Ollama enforces enum constraints at the structured-output
    level — the model literally cannot return an invalid unit value.
    """
    valid_units = UNITS_BY_BILL_TYPE.get(bill_type, UNITS_BY_BILL_TYPE["unknown"])

    return {
        "type": "object",
        "properties": {
            "provider_name":  {"type": ["string", "null"]},
            "customer_name":  {"type": ["string", "null"]},
            "account_number": {"type": ["string", "null"]},
            "bill_date":      {"type": ["string", "null"]},
            "due_date":       {"type": ["string", "null"]},
            "amount_due":     {"type": ["number", "null"]},
            "usage_quantity": {"type": ["number", "null"]},
            "usage_unit": {
                "type": ["string", "null"],
                "enum": valid_units + [None],
            },
            "meter_number":   {"type": ["string", "null"]},
        },
        "required": [
            "provider_name", "customer_name", "account_number",
            "bill_date", "due_date", "amount_due",
            "usage_quantity", "usage_unit", "meter_number",
        ],
        "additionalProperties": False,
    }


def build_user_prompt(bill_text: str, bill_type: str) -> str:
    """
    Leaner version — the schema enum already constrains usage_unit at the
    output level so the prompt only needs a one-line bill type hint.
    """
    valid_units = UNITS_BY_BILL_TYPE.get(bill_type, UNITS_BY_BILL_TYPE["unknown"])
    units_str   = ", ".join(valid_units)

    return f"""Extract these fields from the utility bill text below.

This is a {bill_type} bill — usage_unit must be one of: {units_str}.

FIELDS:

- provider_name (string | null): utility company name from header,
  exactly as printed e.g. "Con Edison", "National Grid".
- customer_name (string | null): account holder's name as written.
- account_number (string | null): preserve exact format including hyphens.
- bill_date (string | null): date bill was issued, YYYY-MM-DD format.
- due_date (string | null): payment due date, YYYY-MM-DD. Labels: "PAY BY",
  "DUE DATE", "Please Pay By". If label and value are on different lines
  due to OCR layout, look 1-2 lines below the label for the date.
- amount_due (number | null)
  Match: the total amount the customer owes for this bill, as a bare number, no $ sign.
  Source labels (in order of preference):
    "TOTAL AMOUNT DUE", "Total amount due", "Amount Due", "AMOUNT DUE",
    "Balance Due", "Please Pay", "Current balance due", "Total Due",
    "Pay This Amount", "PAY THIS AMOUNT"
- usage_quantity (number | null)
  Look for value that near the "Total Usage" text
  Another case is the amount is near the Usage Unit.
  If you know the type of bill then use the Unit as the reference to find the number around that. That is your unit usage.
  The result is right above the text "Total Electric Usage".

- usage_unit (string | null): one of {units_str} for this {bill_type} bill.
- meter_number (string | null): meter identifier as written.
  Do NOT confuse with meter readings.

Reformat any literal date to YYYY-MM-DD. Do NOT invent values.
Return ONLY the JSON object.

BILL TEXT:
\"\"\"
{bill_text}
\"\"\"

Return the JSON object now."""


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_bill_text(file_path: str) -> str:
    """Run the OCR pipeline and return the cleaned text."""
    result    = process_pdf(file_path)
    raw_text  = result["pages"][-1]["full_text"]
    return clean_bill_text(raw_text)


def extract_fields(bill_text: str) -> tuple[dict, str]:
    """
    Detects bill type from OCR text, builds the type-constrained schema
    and prompt, then sends to Ollama.
    """
    bill_type = detect_bill_type(bill_text)
    schema = build_schema(bill_type)
    prompt = build_user_prompt(bill_text, bill_type)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "format": schema,
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 0.1,
                "num_ctx": 8192,
                "seed": 1704,
            },
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return json.loads(response.json()["message"]["content"]), bill_type


def process_one_file(uploaded_file) -> dict:
    """Save upload to a temp path, extract text, run the model. Returns a row dict."""
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    file_bytes = uploaded_file.getvalue()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        bill_text         = extract_bill_text(tmp_path)
        fields, bill_type = extract_fields(bill_text)
        return {
            "source_file":    uploaded_file.name,
            "bill_type":      bill_type,
            "provider_name":  fields.get("provider_name"),
            "customer_name":  fields.get("customer_name"),
            "bill_date":      fields.get("bill_date"),
            "due_date":       fields.get("due_date"),
            "amount_due":     fields.get("amount_due"),
            "account_number": fields.get("account_number"),
            "meter_number":   fields.get("meter_number"),
            "usage_quantity": fields.get("usage_quantity"),
            "usage_unit":     fields.get("usage_unit"),
            "status":         "OK",
            "error":          None,
            "saved_to_db":    False,   # tracks whether user has saved this row
            "save_message":   None,    # last save attempt result (success/skip/error)
            "_bill_text":     bill_text,
            "_file_bytes":    file_bytes,
        }
    except Exception as e:
        return {
            "source_file":    uploaded_file.name,
            "bill_type":      None,
            "provider_name":  None,
            "customer_name":  None,
            "bill_date":      None,
            "due_date":       None,
            "amount_due":     None,
            "account_number": None,
            "meter_number":   None,
            "usage_quantity": None,
            "usage_unit":     None,
            "status":         "ERROR",
            "error":          str(e),
            "saved_to_db":    False,
            "save_message":   None,
            "_bill_text":     "",
            "_file_bytes":    file_bytes,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Database save — wraps app1's row dict into save_bill_to_db's expected shape
# ---------------------------------------------------------------------------

def row_to_db_result(row: dict) -> dict:
    """
    Converts an app1 result row into the format save_bill_to_db expects.

    The app1 row has fields at the top level (customer_name, etc).
    save_bill_to_db expects them inside an "extracted_fields" dict
    along with metadata about the extraction itself.
    """
    extracted = {
        "provider_name":  row.get("provider_name"),
        "customer_name":  row.get("customer_name"),
        "account_number": row.get("account_number"),
        "bill_date":      row.get("bill_date"),
        "due_date":       row.get("due_date"),
        "amount_due":     row.get("amount_due"),
        "usage_quantity": row.get("usage_quantity"),
        "usage_unit":     row.get("usage_unit"),
        "meter_number":   row.get("meter_number"),
    }

    # Compute extraction rate from non-null fields — used for audit reporting
    total_fields  = len(extracted)
    filled_fields = sum(1 for v in extracted.values() if v is not None)
    rate          = f"{filled_fields * 100 // total_fields}%"

    # Flag any nulls as needing review
    low_conf = [k for k, v in extracted.items() if v is None]

    return {
        "source_file":           row["source_file"],
        "extracted_fields":      extracted,
        "extraction_rate":       rate,
        "low_confidence_fields": low_conf,
        "model_used":            f"OCR + {MODEL_NAME}",
    }


def save_row_to_database(row: dict, db_path: str = DB_PATH) -> tuple[bool, str]:
    """
    Saves a single row to the database. Returns (success, message)
    where message describes the outcome for display in the UI.

    Three possible outcomes:
      - Saved          → True,  "Saved as bill_id=N"
      - Duplicate      → False, "Already in database (account+date match)"
      - Error          → False, "Save failed: <error>"

    save_bill_to_db handles auto-creation of provider and customer rows,
    duplicate detection by (account_number, bill_date), date normalisation,
    and amount parsing — we just give it the right input shape.
    """
    if row["status"] != "OK":
        return False, "Cannot save — extraction failed"

    try:
        db_result = row_to_db_result(row)
        saved = save_bill_to_db(db_result, db_path=db_path)

        if saved:
            return True, "✅ Saved to database"

        # Returned False — figure out why by checking the database
        acct = row.get("account_number")
        bill_date = row.get("bill_date")
        if acct and bill_date and Path(db_path).exists():
            try:
                conn = sqlite3.connect(db_path)
                cur = conn.execute(
                    "SELECT bill_id FROM bills "
                    "WHERE account_number = ? AND bill_date = ?",
                    (acct, bill_date)
                )
                existing = cur.fetchone()
                conn.close()
                if existing:
                    return False, (
                        f"⏭ Already in database "
                        f"(bill_id={existing[0]})"
                    )
            except sqlite3.Error:
                pass

        # Wasn't a duplicate — must have been an error caught by save_bill_to_db
        return False, "❌ Save failed — check terminal for details"

    except Exception as e:
        return False, f"❌ Save error: {e}"


def render_original(file_bytes: bytes, file_name: str, max_pages: int = 5):
    """Return a list of PIL images for displaying the original document."""
    suffix = Path(file_name).suffix.lower()

    if suffix in {".png", ".jpg", ".jpeg"}:
        return [Image.open(io.BytesIO(file_bytes))]

    if suffix == ".pdf":
        if not HAS_PYMUPDF:
            return None
        images = []
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc[:max_pages]:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                images.append(Image.open(io.BytesIO(pix.tobytes("png"))))
        return images

    return []


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Bill Field Extractor", page_icon="📄", layout="wide")

st.title("📄 Bill Field Extractor")
st.caption("Upload one or more utility bills (PDF or image). Extracted fields appear in the table below.")

with st.sidebar:
    st.subheader("Settings")
    st.text_input("Ollama URL", value=OLLAMA_URL, key="ollama_url", disabled=True)
    st.text_input("Model", value=MODEL_NAME, key="model_name", disabled=True)
    st.text_input("Database path", value=DB_PATH, key="db_path_display", disabled=True)
    st.markdown("---")
    st.markdown("**Tip:** make sure `ollama serve` is running and the model is pulled.")
    st.markdown("---")
    st.markdown(
        "**Bill type detection:** the OCR text is scanned for keywords "
        "to identify whether the bill is water, electric, gas, or "
        "National Grid. The detected type constrains which usage units "
        "the model is allowed to return — preventing impossible answers "
        "like `kWh` on a water bill."
    )
    st.markdown("---")
    st.markdown(
        "**Database saves:** extractions are NOT saved automatically. "
        "Review each bill, then use the **Save** buttons to commit "
        "individual bills or the whole batch to the database."
    )

uploaded_files = st.file_uploader(
    "Drop bills here",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if "results" not in st.session_state:
    st.session_state.results = None

col_btn1, col_btn2, _ = st.columns([1, 1, 3])
with col_btn1:
    extract_clicked = st.button(
        "🚀 Extract fields",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )
with col_btn2:
    clear_clicked = st.button(
        "🗑️ Clear results",
        disabled=st.session_state.results is None,
        use_container_width=True,
    )

if clear_clicked:
    st.session_state.results = None
    st.rerun()

if extract_clicked and uploaded_files:
    rows = []
    progress = st.progress(0.0, text="Starting...")
    for i, f in enumerate(uploaded_files, start=1):
        progress.progress(
            (i - 1) / len(uploaded_files),
            text=f"Processing {f.name} ({i}/{len(uploaded_files)})",
        )
        rows.append(process_one_file(f))
    progress.progress(1.0, text="Done")
    progress.empty()
    st.session_state.results = rows

if st.session_state.results:
    rows = st.session_state.results
    df = pd.DataFrame(rows)

    total       = len(df)
    ok          = (df["status"] == "OK").sum()
    failed      = total - ok
    saved_count = (df["saved_to_db"] == True).sum() if "saved_to_db" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total bills", total)
    col2.metric("Extracted", ok)
    col3.metric("Failed", failed)
    col4.metric("Saved to DB", saved_count)

    st.subheader("Extracted records")
    visible_cols = [
        "source_file", "bill_type", "provider_name", "customer_name",
        "bill_date", "due_date", "amount_due", "account_number",
        "meter_number", "usage_quantity", "usage_unit", "status",
        "saved_to_db",
    ]
    st.dataframe(df[visible_cols], use_container_width=True, hide_index=True)

    csv_bytes = df[visible_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download as CSV",
        data=csv_bytes,
        file_name="extracted_bills.csv",
        mime="text/csv",
    )

    # ── Bulk save section ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Save to Database")

    # Count how many rows are eligible for saving
    unsaved_ok = [
        r for r in rows
        if r["status"] == "OK" and not r.get("saved_to_db")
    ]
    saved_already = [r for r in rows if r.get("saved_to_db")]

    col_bulk1, col_bulk2 = st.columns([2, 1])

    with col_bulk1:
        st.caption(
            f"**{len(unsaved_ok)} bill(s) ready to save** "
            f"({len(saved_already)} already saved this session). "
            f"Saves use `db_handler.save_bill_to_db` — same logic as the "
            f"Gemini app — so duplicates by (account, bill_date) are "
            f"automatically rejected."
        )

    with col_bulk2:
        if st.button(
            f"💾 Save all {len(unsaved_ok)} unsaved bill(s)",
            type="primary",
            disabled=len(unsaved_ok) == 0,
            use_container_width=True,
            key="save_all_btn",
        ):
            saved_n   = 0
            duplicate = 0
            errors    = 0

            for r in rows:
                if r["status"] != "OK" or r.get("saved_to_db"):
                    continue
                success, message = save_row_to_database(r)
                r["save_message"] = message
                if success:
                    r["saved_to_db"] = True
                    saved_n += 1
                elif "Already" in message or "duplicate" in message.lower():
                    duplicate += 1
                else:
                    errors += 1

            # Update session state so the change persists across reruns
            st.session_state.results = rows

            if saved_n > 0:
                st.success(
                    f"✅ Saved {saved_n} bill(s). "
                    f"{duplicate} duplicate(s) skipped, {errors} error(s)."
                )
            elif duplicate > 0 and errors == 0:
                st.info(
                    f"ℹ️ All {duplicate} bill(s) were already in the database."
                )
            else:
                st.error(
                    f"❌ {errors} save error(s) occurred — check terminal for details."
                )
            st.rerun()

    # ── Per-file detail expander ──────────────────────────────────────────
    with st.expander("🔍 Per-file details — original vs. extracted"):
        for idx, row in enumerate(rows):
            saved_indicator = " · 💾 saved" if row.get("saved_to_db") else ""
            st.markdown(
                f"### {row['source_file']} — `{row['status']}`  ·  "
                f"detected: `{row.get('bill_type', '?')}`{saved_indicator}"
            )

            if row["error"]:
                st.error(row["error"])

            # If we tried to save this row earlier, show the result here
            if row.get("save_message"):
                if row.get("saved_to_db"):
                    st.success(row["save_message"])
                elif "Already" in row["save_message"]:
                    st.info(row["save_message"])
                else:
                    st.warning(row["save_message"])

            col_original, col_extracted = st.columns(2)

            with col_original:
                st.markdown("**Original document**")
                images = render_original(row["_file_bytes"], row["source_file"])

                if images is None:
                    st.info(
                        "PDF preview requires PyMuPDF. Install it with "
                        "`pip install pymupdf` to see the original here."
                    )
                elif images:
                    for page_idx, img in enumerate(images, start=1):
                        if len(images) > 1:
                            st.caption(f"Page {page_idx}")
                        st.image(img, use_container_width=True)
                else:
                    st.warning("Unsupported file type for preview.")

                st.download_button(
                    "⬇️ Download original",
                    data=row["_file_bytes"],
                    file_name=row["source_file"],
                    key=f"dl_{idx}_{row['source_file']}",
                )

            with col_extracted:
                st.markdown("**Extracted text**")
                if row["_bill_text"]:
                    st.text_area(
                        "Extracted text",
                        value=row["_bill_text"],
                        height=400,
                        key=f"text_{idx}_{row['source_file']}",
                        label_visibility="collapsed",
                    )
                else:
                    st.caption("(no text extracted)")

                st.markdown("**Extracted fields**")
                amount = row["amount_due"]
                usage_str = (
                    f"{row['usage_quantity']} {row['usage_unit']}"
                    if row.get("usage_quantity") is not None
                    else "—"
                )
                fields_df = pd.DataFrame({
                    "Field": [
                        "Bill type (detected)", "Provider", "Customer name",
                        "Bill date", "Due date", "Amount due",
                        "Account number", "Meter number", "Usage",
                    ],
                    "Value": [
                        row.get("bill_type") or "—",
                        row.get("provider_name") or "—",
                        row.get("customer_name") or "—",
                        row.get("bill_date") or "—",
                        row.get("due_date") or "—",
                        f"${amount:.2f}" if amount is not None else "—",
                        row.get("account_number") or "—",
                        row.get("meter_number") or "—",
                        usage_str,
                    ],
                })
                st.table(fields_df)

                # Per-bill save button — disabled if extraction failed
                # or if this row has already been saved this session
                if row["status"] == "OK":
                    already_saved = row.get("saved_to_db", False)
                    btn_label = (
                        "💾 Already saved" if already_saved
                        else "💾 Save this bill to database"
                    )
                    if st.button(
                        btn_label,
                        key=f"save_{idx}_{row['source_file']}",
                        disabled=already_saved,
                        use_container_width=True,
                    ):
                        success, message = save_row_to_database(row)
                        row["save_message"] = message
                        if success:
                            row["saved_to_db"] = True
                        # Persist across rerun
                        st.session_state.results = rows
                        st.rerun()

            st.markdown("---")
elif uploaded_files:
    st.info(
        f"📥 {len(uploaded_files)} file(s) ready. "
        "Click **🚀 Extract fields** above to process."
    )
else:
    st.info("Upload one or more bill files to begin.")