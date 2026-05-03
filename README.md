# TroyBanks Bill Extraction Pipeline
An automated utility bill data extraction and querying system built for Troy & Banks audit operations. The pipeline replaces manual auditor workflows with AI-powered field extraction, supporting both cloud-based (Gemini) and fully local (OCR + Ollama) processing paths.

# 🔥 Overview
Auditors at Troy & Banks process hundreds of utility bills per month, extracting key fields by hand to verify charges and detect anomalies. This system automates that work end-to-end:

1. Upload — drop in PDF, JPG, or PNG bills (single or batch)
2. Extract — pull structured fields using either Gemini Vision or local OCR + LLM
3. Review — verify extracted values against the original bill side-by-side, edit anything that's wrong
4. Save — commit corrected data to a SQLite database with auto-managed customer and provider records
5. Query — ask plain-English questions or run pre-built analytics over the bills database

The system is designed for auditors who aren't developers — every action is a button click, every error has a plain-English message, and corrections to extracted data can be made directly in the UI.

# 🎯 Result Example (Sample bill)

From unstructure format like invoices -> structure tables

<img width="2216" height="1442" alt="image" src="https://github.com/user-attachments/assets/8fb08ac0-2a47-4d8c-b19b-d96de54216f5" />

# 🎯 Architecture

### User Interface (Streamlit)
  &darr;
------------------------------------------------
### gemini_app.py (Gemini path)
### OCR_Local_Model.py (OCR + LLM)

## &darr;

## Extractor module
### extractor.py
### gemini_app.py (Gemini Vision API)

## OCR pipeline
### pdfplumber
### Teseract


## &darr;

## LLM Processing Layer
### Ollama (llama3.1)
### JSON schema output



## &darr;

## Database Layer
### db_handler.py
### save_bill_to_db
### Lookup providers
### Lookup customers
### validation logic
### duplicate check

## &darr;

## SQLite Database
### troybanks_bills.db
### (customers, bills, providers)

# ⚡ Two Apps, One Database
The project ships with two Streamlit apps that share the same database:

### OCR_Local_Model.py — Local OCR + Ollama Pipeline (Primary)
Fully local, no cloud calls, free to run. Uses Tesseract for OCR and llama3.1 for field extraction. Useful for sensitive bills, offline operation, or when Gemini quotas are exhausted. Roughly 10 seconds per bill on a 16GB Mac.

#### Features:
1. Bill type detection (water / electric / gas / national_grid) with unit constraints
2. Editable extracted fields before saving
3. Per-bill and bulk save buttons
4. Side-by-side bill image and extracted text comparison


### gemini_app.py — Gemini Cloud Pipeline (Backup)
Best for production audit work. Uses Google's Gemini 2.5 Flash to read bill images directly. Most accurate, especially on visually complex bills (dark header bands, tables, mixed columns). Costs roughly $0.0001-$0.001 per bill on the paid tier.
Tabs:

1. 📄 Extract Bills — upload bills, extract fields, see results side-by-side with the original image
2. 📊 Database — browse all saved bills with editable cells, row-level deletion, and search
3. ℹ️ Help — quick-reference documentation

Both apps write to the same troybanks_bills.db file. Bills extracted via either path are deduplicated together.


# 🔥 Key Features

## Side-by-Side Review
1. Every extraction shows the original bill image alongside the extracted fields. Auditors verify values against the source without switching windows.
2. Editable Extraction Results
3. Wrong field values can be fixed inline — type the correct value, the change flows through to the CSV download and the database save automatically.
4. No need to re-run extraction over a typo.
## Manual Save Buttons
1. Extractions don't auto-save. Auditors review first, edit if needed, then commit to the database with explicit save buttons. Two save flows:

2. Per-bill save — review one, save one
3. Bulk save — confident the whole batch is correct, save them all at once


#  🗄️ Database Schema

### TABLE: bills
  * bill_id              INTEGER PK
  * provider_id          INTEGER → providers.provider_id
  * customer_id          INTEGER → customers.customer_id
  * customer_name        TEXT
  * account_number       TEXT
  * bill_date            TEXT (YYYY-MM-DD)
  * due_date             TEXT (YYYY-MM-DD)
  * meter_number         TEXT
  * usage_quantity       REAL
  * usage_unit           TEXT
  * amount_due           REAL
  * source_file          TEXT
  * extraction_date      TEXT
  * extraction_rate      TEXT
  * needs_review         TEXT
  * model_used           TEXT
  * UNIQUE (account_number, bill_date)

### TABLE: customers
  * customer_id     INTEGER PK
  * customer_name   TEXT
  * account_number  TEXT
  * created_at      DATE
  * UNIQUE (customer_name, account_number)

### TABLE: providers
  * provider_id    INTEGER PK
  * provider_name  TEXT UNIQUE
  * bill_type      TEXT (electric / gas / water / dual / unknown)
  * created_at     DATE

# ⚙️ Installation 

## Prerequisites
### Python 3.14
### Tesseract OCR for Local Pipeline
  * macOS — admin password required
  * sudo brew install tesseract
### Ollama for local
  * Install from ollama.com
  * ollama pull llama3.1:latest
### Python dependencies
* pip install -r requirements.txt
* python3.14 -m pip install -r requirements.txt

### One time set-up
1. Create database
   * python3.14 create_db.py
   * This creates troybanks_bills.db with all three tables.
2. Add your Gemini API key (only needed for gemini_app.py):
   * In .env file at project root
   * GEMINI_API_KEY=your_key_here
   * Get a key at
   * https://aistudio.google.com/
3. Running the app
   * Local llama + OCR (recommend)
     1. ollama serve
     2. python3.14 -m streamlit run OCR_Local_Model.py
4. Gemini API path
   * python3.14 -m streamlit run gemini_app.py
  
# 🚀 Typical Workflow

1. Auditor opens OCR_Local_Model.py or gemini_app.py in a browser
2. Drops 10-30 utility bills into the upload area
3. Clicks 🚀 Extract All Bills (each bill took 10s for local and 5s for Gemini)
4. Reviews results — each bill expander shows the original on the left, extracted fields on the right
5. Spots a wrong field — clicks the cell, types the correction
6. Switches to the Database tab — sees the corrected bill saved with the right values, joined with auto-created provider and customer rows
7. Downloads CSV of any view for reporting

All without writing SQL or touching the database directly.


# 🛡️ Privacy & Security

1. Gemini paid tier is recommended for any production data. The free tier may be used for training; the paid tier explicitly is not. https://ai.google.dev/gemini-api/terms#paid-services
2. Local-only mode (OCR_Local_Model.py) keeps every byte on the auditor's machine. No cloud calls of any kind.
3. The .env file containing GEMINI_API_KEY should never be committed to version control.

# 🧠 Known Limitations

* PDF previews require PyMuPDF — without it, PDFs fall back to a download link rather than inline preview
* Multi-page bills show only the first 3 pages by default in the side-by-side view
* OCR + Ollama path is meaningfully slower than Gemini (~10s vs ~3s per bill on a 16GB Mac)

# 🗄️ Updating the field schema
### If you want to extract additional fields:

* Add the field to BILL_SCHEMA in gemini_client.py
* Add the field to the prompt in the same file
* Add the field to ALL_FIELDS in validate_and_clean
* Update save_bill_to_db in db_handler.py to handle the new field
* Add a column to the bills table in create_db.py (existing databases need a migration via ALTER TABLE)

# 🗄️ Backups
The entire database is one file: troybanks_bills.db. Copy it to back up; restore by replacing the file. SQLite handles the rest.

# 🧠 Acknowledgements
Built as part of a University at Buffalo consulting project for Troy & Banks. Uses Google Gemini for cloud extraction, Meta's llama3.1 via Ollama for local extraction, and Tesseract for OCR.
