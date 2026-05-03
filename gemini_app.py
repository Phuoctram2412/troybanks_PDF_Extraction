"""
TroyBanks Bill Extraction UI
==============================
Run with:  streamlit run app.py
"""

import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()   # load GEMINI_API_KEY from .env before importing extractor

from extractor.extractor import extract_bill, MODEL
from database.db_handler import save_bill_to_db


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = "troybanks_bills.db"


# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be the first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TroyBanks Bill Extractor",
    page_icon="🏦",
    layout="wide"
)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — settings
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    model = st.selectbox(
        "Gemini Model",
        options=["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
        index=0,
        help="Flash = cheapest. Pro = most accurate on complex bills."
    )

    dpi = st.slider(
        "Image DPI",
        min_value=100,
        max_value=300,
        value=150,
        step=50,
        help="Higher DPI = better OCR on small text, but slower and more tokens."
    )

    confidence_threshold = st.slider(
        "Confidence threshold for review",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
        help="Fields below this score will be flagged for manual review."
    )

    st.divider()
    st.caption("TroyBanks PDF Extraction v2.0")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def confidence_badge(level: str) -> str:
    """Returns a coloured emoji badge for a confidence level."""
    return {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴", "UNKNOWN": "⚪"}.get(level, "⚪")


def render_extraction_table(result: dict):
    """
    Renders the extracted fields as a Streamlit dataframe with
    colour coding based on confidence level.
    """
    fields    = result.get("extracted_fields", {})
    conf_data = result.get("confidence", {})

    rows = []
    for field, value in fields.items():
        conf  = conf_data.get(field, {})
        score = conf.get("score")
        level = conf.get("level", "UNKNOWN")
        rows.append({
            "Field":      field,
            "Value":      str(value) if value is not None else "—",
            "Confidence": f"{score:.2f}" if score is not None else "N/A",
            "Status":     f"{confidence_badge(level)} {level}",
        })

    df = pd.DataFrame(rows)

    def highlight_low(row):
        if "🔴" in row["Status"] or "⚪" in row["Status"]:
            return ["background-color: #fff0f0"] * len(row)
        elif "🟡" in row["Status"]:
            return ["background-color: #fffbe6"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df.style.apply(highlight_low, axis=1),
        use_container_width=True,
        hide_index=True,
        height=500
    )


def is_already_in_database(source_file: str, db_path: str = DB_PATH) -> bool:
    """
    Checks whether a bill from this source file is already in the database.
    Used as a quick pre-check before re-extracting — saves Gemini tokens.

    Note: save_bill_to_db ALSO checks for duplicates by (account_number,
    bill_date) which is more reliable than filename matching. This function
    is just an optimisation to skip the API call when we recognise the
    filename. The real duplicate guarantee comes from the database.
    """
    if not Path(db_path).exists():
        return False
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.execute(
            "SELECT 1 FROM bills WHERE source_file = ? LIMIT 1",
            (source_file,)
        )
        found = cur.fetchone() is not None
        conn.close()
        return found
    except sqlite3.Error:
        # If the table doesn't exist yet or any other DB issue, treat as
        # not-found — save_bill_to_db will create the table on first save
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📄 Extract Bills", "📊 Database", "ℹ️ Help"])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Upload and Extract
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.header("📄 Bill Extraction")
    st.caption("Upload one or more utility bills to extract fields using Gemini AI.")

    uploaded_files = st.file_uploader(
        "Drop bills here",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Accepts PDF, JPG, or PNG. Multiple files are processed as a batch."
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) ready to process.")

        if st.button("🚀 Extract All Bills", type="primary", use_container_width=True):

            tmp_dir = Path("tmp_uploads")
            tmp_dir.mkdir(exist_ok=True)

            all_results = []
            progress    = st.progress(0, text="Starting...")
            status_box  = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                name     = uploaded_file.name
                tmp_path = tmp_dir / name

                progress.progress(
                    i / len(uploaded_files),
                    text=f"Processing {name} ({i+1}/{len(uploaded_files)})..."
                )

                # ── Pre-check: skip if filename already in database ──────────
                # This saves a Gemini API call on obvious re-uploads.
                # save_bill_to_db has a more reliable check based on
                # (account_number, bill_date) which catches duplicates even
                # if the filename was different.
                if is_already_in_database(name):
                    status_box.info(f"⏭ {name} already in database — skipping")
                    all_results.append({
                        "source_file":           name,
                        "extracted_fields":      {},
                        "confidence":            {},
                        "extraction_rate":       "skipped",
                        "low_confidence_fields": [],
                        "skipped":               True,
                        "skip_reason":           "Filename already in database",
                    })
                    continue

                status_box.info(f"Processing {name}...")
                tmp_path.write_bytes(uploaded_file.getvalue())

                try:
                    # Run Gemini extraction
                    result = extract_bill(str(tmp_path), model=model, dpi=dpi)

                    # Save via db_handler.save_bill_to_db — handles:
                    #   - duplicate detection by (account_number, bill_date)
                    #   - provider auto-creation
                    #   - customer auto-creation
                    #   - date normalisation
                    #   - amount parsing
                    #   - all 21 column inserts in correct order
                    saved = save_bill_to_db(result, db_path=DB_PATH)

                    if saved:
                        status_box.success(f"✅ {name} extracted and saved")
                    else:
                        # Returned False — meaning duplicate caught at DB level.
                        # The bill was extracted (Gemini call already made) but
                        # not saved because account+date already existed.
                        status_box.warning(
                            f"⏭ {name} — duplicate by account+date, not saved"
                        )
                        result["skipped"]     = True
                        result["skip_reason"] = "Duplicate by (account, customer_name)"

                    all_results.append(result)

                except RuntimeError as e:
                    st.error(f"❌ {name}: {e}")
                    all_results.append({
                        "source_file":           name,
                        "extraction_error":      str(e),
                        "extracted_fields":      {},
                        "confidence":            {},
                        "extraction_rate":       "0%",
                        "low_confidence_fields": [],
                    })

                # Pause between calls — remove once on the paid Gemini tier
                if i < len(uploaded_files) - 1:
                    time.sleep(4)

            progress.progress(1.0, text="Done!")

            saved_count   = sum(
                1 for r in all_results
                if not r.get("extraction_error") and not r.get("skipped")
            )
            skipped_count = sum(1 for r in all_results if r.get("skipped"))
            failed_count  = sum(1 for r in all_results if r.get("extraction_error"))

            status_box.success(
                f"✅ Done — {saved_count} saved, "
                f"{skipped_count} skipped, "
                f"{failed_count} failed"
            )

            st.session_state["results"] = all_results

        # ── Display results ───────────────────────────────────────────────
        if "results" in st.session_state:
            results = st.session_state["results"]

            successful    = [
                r for r in results
                if not r.get("extraction_error") and not r.get("skipped")
            ]
            needs_review  = [r for r in successful if r.get("low_confidence_fields")]
            failed        = [r for r in results if r.get("extraction_error")]
            skipped       = [r for r in results if r.get("skipped")]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Bills Processed", len(results))
            col2.metric("Saved",           len(successful))
            col3.metric("Need Review",     len(needs_review))
            col4.metric("Failed",          len(failed))

            if skipped:
                st.info(
                    f"⏭ {len(skipped)} bill(s) were already in the database "
                    f"and skipped to avoid duplicates."
                )

            st.divider()

            for result in results:
                name  = result.get("source_file", "unknown")
                rate  = result.get("extraction_rate", "0%")
                error = result.get("extraction_error")
                skip  = result.get("skipped")

                if error:
                    label = f"❌ {name}  —  extraction failed"
                elif skip:
                    label = f"⏭ {name}  —  {result.get('skip_reason', 'already in database')}"
                elif result.get("low_confidence_fields"):
                    label = f"⚠️ {name}  —  {rate} extracted (review needed)"
                else:
                    label = f"✅ {name}  —  {rate} extracted"

                with st.expander(label, expanded=False):
                    if error:
                        st.error(f"Extraction failed: {error}")
                    elif skip:
                        st.info(
                            f"This bill was not saved: "
                            f"{result.get('skip_reason', 'already in database')}"
                        )
                    else:
                        if result.get("low_confidence_fields"):
                            st.warning(
                                f"🔴 Review these fields manually: "
                                f"{', '.join(result['low_confidence_fields'])}"
                            )

                        render_extraction_table(result)

                        st.download_button(
                            label="⬇️ Download JSON",
                            data=json.dumps(result["extracted_fields"], indent=2),
                            file_name=f"{Path(name).stem}_extracted.json",
                            mime="application/json",
                            key=f"dl_{name}"
                        )

            if successful:
                all_records = [
                    {"source_file": r["source_file"], **r["extracted_fields"]}
                    for r in successful
                ]
                csv = pd.DataFrame(all_records).to_csv(index=False)
                st.download_button(
                    label="⬇️ Download All as CSV",
                    data=csv,
                    file_name="extracted_bills.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Database viewer
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.header("📊 Bills Database")

    if not Path(DB_PATH).exists():
        st.info("No database yet — process some bills in the Extract tab first.")
    else:
        conn = sqlite3.connect(DB_PATH)

        try:
            # Join with providers to show provider name instead of just ID
            df = pd.read_sql("""
                SELECT b.bill_id,
                       COALESCE(p.provider_name, 'Unknown') AS provider,
                       b.customer_name,
                       b.account_number,
                       b.bill_date,
                       b.due_date,
                       b.amount_due,
                       b.usage_quantity,
                       b.usage_unit,
                       b.source_file
                FROM   bills b
                LEFT JOIN providers p ON b.provider_id = p.provider_id
                ORDER BY b.bill_date DESC
            """, conn)

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Bills",   len(df))

            if "amount_due" in df.columns:
                total = pd.to_numeric(df["amount_due"], errors="coerce").sum()
                col2.metric("Total Amount Due", f"${total:,.2f}")

            if "extraction_rate" in df.columns:
                rates = df["extraction_rate"].dropna()
                rates = rates[rates != "skipped"]
                if not rates.empty:
                    avg_rate = rates.str.rstrip("%").astype(float).mean()
                    col3.metric("Avg Extraction Rate", f"{avg_rate:.0f}%")

            st.divider()

            search = st.text_input("🔍 Search by customer name or account number")
            if search:
                mask = (
                    df["customer_name"].str.contains(search, case=False, na=False) |
                    df["account_number"].str.contains(search, case=False, na=False)
                )
                df = df[mask]
                st.caption(f"{len(df)} matching record(s)")

            st.dataframe(df, use_container_width=True, hide_index=True)

            st.download_button(
                label="⬇️ Download Full Database as CSV",
                data=df.to_csv(index=False),
                file_name="troybanks_bills_database.csv",
                mime="text/csv",
                use_container_width=True
            )

        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Help
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.header("ℹ️ How to Use")

    st.markdown("""
    ### Quick Start
    1. Go to the **Extract Bills** tab
    2. Drag and drop your utility bills (PDF, JPG, or PNG)
    3. Click **Extract All Bills**
    4. Review the results — 🔴 red rows need manual verification
    5. Download the results as JSON or CSV

    ### Duplicate Detection
    Bills are deduplicated in two ways:
    - **By filename** (quick check before extraction — saves Gemini tokens)
    - **By account number + bill date** (reliable check at the database level)

    Re-uploading the same bill — even with a different filename — is safe.
    The database will reject it as a duplicate after extraction.

    ### Confidence Levels
    | Badge | Level | Meaning |
    |-------|-------|---------|
    | 🟢 | HIGH | Field clearly visible — safe to use |
    | 🟡 | MEDIUM | Slightly uncertain — spot-check recommended |
    | 🔴 | LOW | Unclear or inferred — manual review required |
    | ⚪ | UNKNOWN | Field not found on this bill |

    ### Model Selection
    | Model | Best For | Cost |
    |-------|----------|------|
    | gemini-2.5-flash | Most bills — best balance | ~$0.15/M tokens |
    | gemini-2.5-flash-lite | Simple clean bills | ~$0.10/M tokens |
    | gemini-2.5-pro | Complex or messy scans | ~$2.00/M tokens |

    ### Rate Limits
    - **Free tier**: 20 requests/day — enough for testing
    - **Paid tier**: 2,000 requests/minute — production ready
    - Add billing at [aistudio.google.com](https://aistudio.google.com)

    ### Supported Bill Types
    - ✅ Washington Water Service
    - ✅ Power Energy (Electric)
    - ✅ National Grid
    - ✅ Any utility bill — Gemini reads them all
    """)