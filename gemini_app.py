"""
TroyBanks Bill Extraction UI
==============================
Run with:  streamlit run app.py
"""

import io
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()   # load GEMINI_API_KEY from .env before importing extractor

from extractor import extract_bill, MODEL
from db_handler import save_bill_to_db

# Optional PDF rendering — if not installed, PDFs fall back to a placeholder
# message and only the extracted fields show. Image files (PNG/JPG) work
# without PyMuPDF since PIL handles them natively.
try:
    import fitz   # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = "troybanks_bills.db"
TMP_UPLOAD_DIR = Path("tmp_uploads")


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


def render_extraction_table(result: dict, height: int = 500):
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
        height=height,
    )


def render_bill_preview(file_path: str, max_pages: int = 3):
    """
    Renders the original bill image alongside the extracted fields.

    Behaviour by file type:
      - PNG/JPG/JPEG : opens directly with PIL — always works
      - PDF          : renders each page (up to max_pages) using PyMuPDF
                       at 2x zoom for legibility. Falls back to a download
                       button if PyMuPDF isn't installed.
      - missing file : shows an info message (the temp file may have been
                       cleared between sessions — reupload to see the bill)

    Why this lives here instead of in the extractor:
      The extractor only handles the file once during extraction. By the
      time the auditor opens the expander to review, the file may have
      been deleted from tmp_uploads. We re-render here on demand.
    """
    path = Path(file_path)

    if not path.exists():
        st.info(
            "📂 Original bill not available for preview. "
            "Re-upload the file to see it here."
        )
        return

    suffix = path.suffix.lower()

    try:
        if suffix in {".png", ".jpg", ".jpeg"}:
            # Image file — load directly with PIL
            img = Image.open(path)
            st.image(img, use_container_width=True, caption=path.name)
            return

        if suffix == ".pdf":
            if not HAS_PYMUPDF:
                st.warning(
                    "PDF preview requires PyMuPDF. "
                    "Run `pip install pymupdf` to enable bill image previews."
                )
                with open(path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF instead",
                        data=f.read(),
                        file_name=path.name,
                        mime="application/pdf",
                        key=f"dl_pdf_{path.name}",
                    )
                return

            # Render each page at 2x for legibility — capped at max_pages
            # so a long bill doesn't dominate the screen
            with fitz.open(str(path)) as doc:
                pages_to_show = min(len(doc), max_pages)
                if len(doc) > max_pages:
                    st.caption(
                        f"Showing first {max_pages} of {len(doc)} pages — "
                        f"download the original to see all pages."
                    )
                for page_idx in range(pages_to_show):
                    page = doc[page_idx]
                    pix  = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img  = Image.open(io.BytesIO(pix.tobytes("png")))
                    if pages_to_show > 1:
                        st.caption(f"Page {page_idx + 1}")
                    st.image(img, use_container_width=True)
            return

        st.warning(f"Preview not supported for {suffix} files.")

    except Exception as e:
        st.warning(f"Could not render preview: {e}")


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

            TMP_UPLOAD_DIR.mkdir(exist_ok=True)

            all_results = []
            progress    = st.progress(0, text="Starting...")
            status_box  = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                name     = uploaded_file.name
                tmp_path = TMP_UPLOAD_DIR / name

                progress.progress(
                    i / len(uploaded_files),
                    text=f"Processing {name} ({i+1}/{len(uploaded_files)})..."
                )

                # ── Pre-check: skip if filename already in database ──────────
                if is_already_in_database(name):
                    status_box.info(f"⏭ {name} already in database — skipping")
                    all_results.append({
                        "source_file":           name,
                        "tmp_path":              str(tmp_path),
                        "extracted_fields":      {},
                        "confidence":            {},
                        "extraction_rate":       "skipped",
                        "low_confidence_fields": [],
                        "skipped":               True,
                        "skip_reason":           "Filename already in database",
                    })
                    # Save the file anyway so the user can preview it
                    # if they want to verify the duplicate
                    tmp_path.write_bytes(uploaded_file.getvalue())
                    continue

                status_box.info(f"Processing {name}...")
                tmp_path.write_bytes(uploaded_file.getvalue())

                try:
                    result = extract_bill(str(tmp_path), model=model, dpi=dpi)
                    # Track tmp_path so the expander can show the bill image
                    # alongside the extracted fields
                    result["tmp_path"] = str(tmp_path)

                    saved = save_bill_to_db(result, db_path=DB_PATH)

                    if saved:
                        status_box.success(f"✅ {name} extracted and saved")
                    else:
                        status_box.warning(
                            f"⏭ {name} — duplicate by account+date, not saved"
                        )
                        result["skipped"]     = True
                        result["skip_reason"] = "Duplicate by (account, bill_date)"

                    all_results.append(result)

                except RuntimeError as e:
                    st.error(f"❌ {name}: {e}")
                    all_results.append({
                        "source_file":           name,
                        "tmp_path":              str(tmp_path),
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
                tmp_path = result.get("tmp_path")

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
                        # Even on error, show the bill so the auditor can
                        # see what went wrong
                        if tmp_path:
                            st.markdown("**Original Bill**")
                            render_bill_preview(tmp_path)

                    elif skip:
                        st.info(
                            f"This bill was not saved: "
                            f"{result.get('skip_reason', 'already in database')}"
                        )
                        # Show the bill so auditor can verify it really is
                        # the same as what's already in the database
                        if tmp_path:
                            render_bill_preview(tmp_path)

                    else:
                        if result.get("low_confidence_fields"):
                            st.warning(
                                f"🔴 Review these fields manually: "
                                f"{', '.join(result['low_confidence_fields'])}"
                            )

                        # Side-by-side: bill image on left, extracted fields
                        # on right. Column ratio 1:1.4 gives the table a bit
                        # more horizontal space for the wider data columns.
                        col_bill, col_fields = st.columns([1, 1.4])

                        with col_bill:
                            st.markdown("**📄 Original Bill**")
                            if tmp_path:
                                render_bill_preview(tmp_path)
                            else:
                                st.info("Bill image not available for preview.")

                        with col_fields:
                            st.markdown("**📋 Extracted Fields**")
                            render_extraction_table(result, height=500)

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
            # Add a "Select" boolean column for the deletion workflow.
            # The column has to come from somewhere when we render the
            # data_editor — we add it after fetching from the database
            # and never persist it back. It exists only in the UI session.
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

            # ── Search ────────────────────────────────────────────────────
            search = st.text_input("🔍 Search by customer name or account number")
            if search:
                mask = (
                    df["customer_name"].str.contains(search, case=False, na=False) |
                    df["account_number"].str.contains(search, case=False, na=False)
                )
                df = df[mask]
                st.caption(f"{len(df)} matching record(s)")

            # ── Editable + selectable table ───────────────────────────────
            # Add a "Select" boolean column for the deletion workflow.
            # It defaults to False on every row and never gets persisted —
            # it lives only in the UI session.
            df_edit = df.copy()
            df_edit.insert(0, "Select", False)

            # Build column configuration:
            #   - Select         : checkbox column for deletion selection
            #   - bill_id        : disabled — primary key, never editable
            #   - provider       : disabled — derived from JOIN, edit via providers table
            #   - source_file    : disabled — set at extraction time
            #   - everything else: editable
            #
            # The auditor can fix any extracted value (a wrong amount_due,
            # a misread account number) without re-running extraction.
            column_config = {
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Tick to mark this bill for deletion",
                    default=False,
                    width="small",
                ),
                "bill_id":     st.column_config.NumberColumn(
                    "Bill ID", disabled=True, width="small"
                ),
                "provider":    st.column_config.TextColumn(
                    "Provider", disabled=True,
                    help="Read-only — provider is auto-detected from extraction"
                ),
                "source_file": st.column_config.TextColumn(
                    "Source File", disabled=True,
                    help="Read-only — set at extraction time"
                ),
                # Editable columns get type-appropriate widgets so the
                # auditor can't put a string in amount_due, etc.
                "customer_name":   st.column_config.TextColumn("Customer"),
                "account_number":  st.column_config.TextColumn("Account #"),
                "bill_date":       st.column_config.TextColumn(
                    "Bill Date",
                    help="YYYY-MM-DD format",
                ),
                "due_date":        st.column_config.TextColumn(
                    "Due Date",
                    help="YYYY-MM-DD format",
                ),
                "amount_due":      st.column_config.NumberColumn(
                    "Amount Due", format="$%.2f"
                ),
                "usage_quantity":  st.column_config.NumberColumn("Usage"),
                "usage_unit":      st.column_config.SelectboxColumn(
                    "Unit",
                    options=["kWh", "CF", "CCF", "therms", "gallons", None],
                ),
            }

            st.caption(
                "💡 **Tip:** Click any cell to edit it. "
                "Tick the **Select** checkbox to mark rows for deletion. "
                "Changes are previewed in the table — click **Save Changes** "
                "or **Delete Selected** to apply them to the database."
            )

            edited_df = st.data_editor(
                df_edit,
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",   # don't allow adding new rows from the UI
                key="bills_editor",
            )

            # ── Detect what changed ───────────────────────────────────────
            # Compare edited_df (what the user sees) with df (database state)
            # to find:
            #   1. Rows where any field other than "Select" was modified
            #   2. Rows where Select is True (marked for deletion)
            #
            # The "Select" column doesn't exist in df, so we drop it before
            # comparing for value changes.
            EDITABLE_COLS = [
                "customer_name", "account_number", "bill_date", "due_date",
                "amount_due", "usage_quantity", "usage_unit",
            ]

            # Find rows with edited values
            edited_values = edited_df.drop(columns=["Select"])
            changed_rows  = []
            for _, edit_row in edited_values.iterrows():
                bill_id = edit_row["bill_id"]
                orig_row = df[df["bill_id"] == bill_id].iloc[0]
                # Find which fields differ
                diffs = {}
                for col in EDITABLE_COLS:
                    orig_val = orig_row[col]
                    new_val  = edit_row[col]
                    # Treat NaN/None as equal to NaN/None
                    orig_is_null = pd.isna(orig_val)
                    new_is_null  = pd.isna(new_val)
                    if orig_is_null and new_is_null:
                        continue
                    if orig_is_null != new_is_null or orig_val != new_val:
                        diffs[col] = new_val
                if diffs:
                    changed_rows.append((int(bill_id), diffs))

            # Find rows marked for deletion
            selected_ids = edited_df.loc[edited_df["Select"] == True, "bill_id"].tolist()
            selected_ids = [int(x) for x in selected_ids]

            # ── Action buttons ────────────────────────────────────────────
            col_save, col_delete, col_dl = st.columns([1, 1, 1])

            with col_save:
                save_disabled = len(changed_rows) == 0
                save_label    = (
                    f"💾 Save {len(changed_rows)} edit(s)"
                    if changed_rows else
                    "💾 Save Changes"
                )
                if st.button(
                    save_label,
                    type="primary",
                    disabled=save_disabled,
                    use_container_width=True,
                    key="save_edits_btn",
                ):
                    saved_n = 0
                    errors  = 0
                    save_conn = sqlite3.connect(DB_PATH)
                    try:
                        for bill_id, diffs in changed_rows:
                            try:
                                # Build a parameterised UPDATE for just the
                                # changed columns — never touch fields the
                                # user didn't edit
                                set_clause = ", ".join(
                                    f"{col} = ?" for col in diffs.keys()
                                )
                                values = list(diffs.values()) + [bill_id]
                                # Convert NaN to None for SQLite
                                values = [
                                    None if pd.isna(v) else v for v in values
                                ]
                                save_conn.execute(
                                    f"UPDATE bills SET {set_clause} "
                                    f"WHERE bill_id = ?",
                                    values,
                                )
                                saved_n += 1
                            except Exception as e:
                                st.error(
                                    f"❌ Couldn't update bill_id={bill_id}: {e}"
                                )
                                errors += 1
                        save_conn.commit()
                    finally:
                        save_conn.close()

                    if saved_n:
                        st.success(
                            f"✅ Updated {saved_n} bill(s)" +
                            (f" ({errors} error(s))" if errors else "")
                        )
                        st.rerun()
                    elif errors == 0:
                        st.info("No changes to save")

            with col_delete:
                del_disabled = len(selected_ids) == 0
                del_label    = (
                    f"🗑️ Delete {len(selected_ids)} selected"
                    if selected_ids else
                    "🗑️ Delete Selected"
                )

                # Use a confirmation pattern — first click arms the action,
                # second click within the same session executes it. Prevents
                # one-click accidental deletions.
                confirm_key = "confirm_delete_armed"
                if confirm_key not in st.session_state:
                    st.session_state[confirm_key] = False

                if st.button(
                    del_label,
                    disabled=del_disabled,
                    use_container_width=True,
                    key="delete_selected_btn",
                ):
                    if not st.session_state[confirm_key]:
                        # First click — arm and warn
                        st.session_state[confirm_key] = True
                        st.warning(
                            f"⚠️ About to delete {len(selected_ids)} bill(s) "
                            f"(IDs: {selected_ids}). "
                            f"Click **Delete Selected** again to confirm. "
                            f"This cannot be undone."
                        )
                    else:
                        # Second click — execute the delete
                        del_conn = sqlite3.connect(DB_PATH)
                        try:
                            placeholders = ",".join("?" * len(selected_ids))
                            del_conn.execute(
                                f"DELETE FROM bills "
                                f"WHERE bill_id IN ({placeholders})",
                                selected_ids,
                            )
                            del_conn.commit()
                            st.success(
                                f"🗑️ Deleted {len(selected_ids)} bill(s)"
                            )
                        except Exception as e:
                            st.error(f"❌ Delete failed: {e}")
                        finally:
                            del_conn.close()
                        # Reset confirm state for next deletion
                        st.session_state[confirm_key] = False
                        st.rerun()

                # Reset confirm state if user does anything else in the UI
                # (e.g. unticks all rows after arming) so the second click
                # doesn't fire on a different selection
                if del_disabled and st.session_state[confirm_key]:
                    st.session_state[confirm_key] = False

            with col_dl:
                st.download_button(
                    label="⬇️ Download CSV",
                    data=df.to_csv(index=False),
                    file_name="troybanks_bills_database.csv",
                    mime="text/csv",
                    use_container_width=True,
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
    4. Click any expander to review — the original bill appears next to
       the extracted fields so you can verify each value
    5. Download the results as JSON or CSV

    ### Side-by-Side Review
    Each result shows the original bill image on the left and the
    extracted fields on the right. This lets you quickly verify:
    - Field values match what's printed on the bill
    - Low confidence fields (🔴) are correct
    - The right bill was processed (no file mix-ups)

    For PDFs, only the first 3 pages are previewed by default — most
    bills are 1-2 pages, but multi-page bills can be downloaded in full.

    ### Editing & Deleting Records
    Open the **Database** tab to manage saved bills:
    - **Click any cell** in the table to edit its value. Editable fields
      include customer name, account number, dates, amount due, and usage.
      Bill ID, provider, and source file are read-only.
    - **Tick the Select checkbox** on the left to mark rows for deletion,
      then click **Delete Selected**. You'll be asked to confirm before
      anything is removed.
    - Click **Save Changes** after editing to write your corrections to
      the database. The button shows how many edits will be saved.

    Use this to fix any field where extraction got it wrong without
    having to re-run the whole bill through Gemini.

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
