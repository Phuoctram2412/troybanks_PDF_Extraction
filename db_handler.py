import re
import sqlite3
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Date normalisation
# ─────────────────────────────────────────────────────────────────────────────

DATE_FORMATS = [
    "%Y-%m-%d",   # already ISO — pass through
    "%m/%d/%Y",   # 04/13/2021
    "%m/%d/%y",   # 04/13/21
    "%B %d, %Y",  # April 13, 2021
    "%b %d, %Y",  # Apr 13, 2021
    "%b %d %Y",   # Apr 13 2021 (no comma)
    "%m-%d-%Y",   # 04-13-2021
]


def normalise_date(value: str | None) -> str | None:
    """
    Converts any date string to YYYY-MM-DD for SQLite storage.
    ISO 8601 format is required for ORDER BY and WHERE comparisons
    to work correctly in SQLite.
    Returns None if the value can't be parsed.
    """
    if not value or not isinstance(value, str):
        return None

    value = value.strip()

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    print(f"  ⚠ Could not parse date: '{value}' — stored as null")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Customer lookup / auto-create
# ─────────────────────────────────────────────────────────────────────────────

def lookup_or_create_customer(conn: sqlite3.Connection,
                                customer_name: str | None,
                                account_number: str | None = None) -> int | None:
    """
    Looks up the customer by name. If it doesn't exist yet, creates a
    new row in the customers table automatically.

    Strategy (in order):
      1. Exact match on (name, account_number) → return that customer_id
      2. Name-only match → return that customer_id (assumes same person)
         and back-fill account_number if we now have one
      3. No match → INSERT new customer

    Returns customer_id, or None if no name was extracted.

    Why name-only match counts as "same person":
      A landlord paying multiple utility bills may have slightly different
      service addresses on each bill (each property), but their billing
      name stays the same. Treating same-name as same-customer keeps their
      bills grouped without proliferating duplicate customer records.

      If you instead want strict (name + address) uniqueness, remove the
      Strategy 2 fallback below.
    """
    if not customer_name or not customer_name.strip():
        return None

    customer_name = customer_name.strip()
    account_number = account_number.strip() if account_number else None

    cur = conn.cursor()

    # Strategy 1 — exact match on (name, address)
    cur.execute(
        "SELECT customer_id FROM customers "
        "WHERE customer_name = ? AND COALESCE(account_number, '') = COALESCE(?, '')",
        (customer_name, account_number)
    )
    row = cur.fetchone()
    if row:
        return row[0]

    # Strategy 2 — match on name alone (treats same name = same customer)
    cur.execute("SELECT customer_id, account_number FROM customers WHERE customer_name = ?",
                (customer_name,))
    row = cur.fetchone()
    if row:
        existing_id, existing_addr = row
        # Back-fill account_number if we have one and the existing record didn't
        if account_number and not existing_addr:
            cur.execute(
                "UPDATE customers SET account_number = ? WHERE customer_id = ?",
                (account_number, existing_id)
            )
            print(f"  ✦ Updated address for existing customer '{customer_name}'")
        return existing_id

    # Strategy 3 — not found → create new customer
    cur.execute(
        "INSERT INTO customers (customer_name, account_number) VALUES (?, ?)",
        (customer_name, account_number)
    )
    new_id = cur.lastrowid
    print(f"  ✦ New customer registered: '{customer_name}' "
          f"(customer_id={new_id})")
    return new_id


# ─────────────────────────────────────────────────────────────────────────────
# Provider lookup / auto-create
# ─────────────────────────────────────────────────────────────────────────────

def lookup_or_create_provider(conn: sqlite3.Connection,
                               extracted_fields: dict) -> int | None:
    """
    Looks up the provider by name. If it doesn't exist yet, creates a
    new row in the providers table automatically.

    New utility companies are registered on first encounter — no manual
    seeding needed. Returns provider_id, or None if no name was extracted.
    """
    cur           = conn.cursor()
    provider_name = extracted_fields.get("provider_name")

    if not provider_name or not provider_name.strip():
        return None

    provider_name = provider_name.strip()

    # Step 1 — exact match (case-insensitive)
    cur.execute(
        "SELECT provider_id FROM providers WHERE LOWER(provider_name) = LOWER(?)",
        (provider_name,)
    )
    row = cur.fetchone()
    if row:
        return row[0]

    # Step 2 — partial match
    # Handles slight variations e.g. "Con Ed" vs "Con Edison"
    cur.execute("SELECT provider_id, provider_name FROM providers")
    provider_lower = provider_name.lower()
    for provider_id, name in cur.fetchall():
        name_lower = name.lower()
        if name_lower in provider_lower or provider_lower in name_lower:
            print(f"  Matched '{provider_name}' → '{name}' (partial match)")
            return provider_id

    # Step 3 — not found → auto-create
    # Infer bill_type from usage_unit if possible
    usage_unit = (extracted_fields.get("usage_unit") or "").upper()
    if usage_unit == "KWH":
        bill_type = "electric"
    elif usage_unit in ("CF", "CCF", "THERMS"):
        bill_type = "gas"
    elif usage_unit == "GALLONS":
        bill_type = "water"
    else:
        bill_type = "unknown"

    cur.execute("""
        INSERT INTO providers (provider_name, bill_type)
        VALUES (?, ?)
    """, (provider_name, bill_type))

    new_id = cur.lastrowid
    print(f"  ✦ New provider registered: '{provider_name}' "
          f"(provider_id={new_id}, bill_type={bill_type})")
    return new_id


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_usage(total_usage: str | None) -> tuple[float | None, str | None]:
    """
    Splits a usage string into quantity and unit.
    "1414 CF" → (1414.0, "CF"),  "228 kWh" → (228.0, "kWh")
    """
    if not total_usage:
        return None, None

    match = re.match(r"([\d,]+\.?\d*)\s*([a-zA-Z]+)", total_usage.strip())
    if match:
        quantity = float(match.group(1).replace(",", ""))
        unit     = match.group(2).upper()
        unit     = {
            "KWH": "kWh", "CF": "CF", "CCF": "CCF",
            "THERMS": "therms", "THERM": "therms",
            "GALLONS": "gallons", "GALLON": "gallons",
        }.get(unit, unit)
        return quantity, unit

    return None, None


def parse_service_period(service_period: str | None) -> tuple[str | None, str | None]:
    """
    Splits "3/6/21 - 4/6/21" → ("3/6/21", "4/6/21").
    Returns (original, None) if the string can't be split.
    """
    if not service_period:
        return None, None

    for separator in [" - ", " – ", " to ", "-"]:
        parts = service_period.split(separator)
        if len(parts) == 2:
            start, end = parts[0].strip(), parts[1].strip()
            if start and end:
                return start, end

    return service_period, None


def parse_amount(value) -> float | None:
    """
    Converts "$31.90", "-$69.27", "None", 31.90 → float or None.
    "None" is a Con Edison special case meaning $0.00.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    cleaned = str(value).strip()

    if cleaned.lower() == "none":
        return 0.00

    cleaned = re.sub(r"[$,\s]", "", cleaned)
    cleaned = cleaned.replace("(", "-").replace(")", "")

    try:
        return float(cleaned)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main save function
# ─────────────────────────────────────────────────────────────────────────────
def save_bill_to_db(result: dict,
                     db_path: str = "troybanks_bills.db") -> bool:
    """
    Transforms the extraction result into a database row and inserts
    it into the bills table.

    The bill is linked to BOTH a provider_id and a customer_id via
    lookup-or-create lookups. New customers and new providers are
    automatically registered on first encounter.

    The customer_name column is also kept on the bill row as a denormalised
    convenience field — this keeps existing queries working and means the
    Database tab can show the name without a JOIN. The customer_id is the
    source of truth for grouping bills by customer.

    Returns True if saved, False if skipped (duplicate) or failed.
    """
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        fields         = result.get("extracted_fields", {})
        account_number = fields.get("account_number")
        # Normalise the bill date to YYYY-MM-DD before using it for the
        # duplicate check. Without normalisation, "04/13/2021" and
        # "2021-04-13" wouldn't match each other and we'd save duplicates.
        bill_date      = normalise_date(fields.get("bill_date"))

        # Warn if key uniqueness fields are missing — without them, we
        # can't reliably detect duplicates and the bill might be saved
        # multiple times across re-uploads
        if not account_number or not bill_date:
            missing = []
            if not account_number: missing.append("account_number")
            if not bill_date:      missing.append("bill_date")
            print(f"  ⚠ Missing key fields: {missing} — "
                  f"saving anyway but duplicate check may not work")

        # Explicit duplicate check before inserting.
        # The bills table also has a UNIQUE constraint on (account_number,
        # bill_date) at the schema level, so even if this check is bypassed
        # the database itself rejects duplicates. This Python check just
        # gives us a clearer return value and log message.
        if account_number and bill_date:
            cur = conn.cursor()
            cur.execute(
                "SELECT bill_id FROM bills "
                "WHERE account_number = ? AND bill_date = ?",
                (account_number, bill_date)        
            )
            existing = cur.fetchone()
            if existing:
                print(f"  ⏭ Duplicate — account={account_number} "
                      f"date={bill_date} already exists as bill_id={existing[0]}")
                return False

        # Resolve foreign keys — both auto-create on first encounter
        provider_id = lookup_or_create_provider(conn, fields)
        customer_id = lookup_or_create_customer(
            conn,
            fields.get("customer_name"),
            fields.get("account_number")
        )

        # Handle both old field names (service_period, total_usage) and
        # new split field names (service_period_start/end, usage_quantity/unit)
        if "total_usage" in fields:
            usage_quantity, usage_unit = parse_usage(fields.get("total_usage"))
        else:
            usage_quantity = fields.get("usage_quantity")
            usage_unit     = fields.get("usage_unit")

        # if "service_period" in fields:
        #     service_start, service_end = parse_service_period(
        #         fields.get("service_period")
        #     )
        # else:
        #     service_start = fields.get("service_period_start")
        #     service_end   = fields.get("service_period_end")

        row = {
            "provider_id":          provider_id,
            "customer_id":          customer_id,
            # customer_name kept as denormalised field for convenience —
            # source of truth is customers.name via customer_id JOIN
            "customer_name":        fields.get("customer_name"),
            "account_number":       account_number,
            "bill_date":            bill_date,         # ← fixed: was customer_name
            "due_date":             normalise_date(fields.get("due_date")),
            "meter_number":         fields.get("meter_number"),
            # "previous_read_date":   normalise_date(
            #                             fields.get("previous_read_date")
            #                             or fields.get("previous_meter_read_date")
            #                         ),
            # "previous_reading":     fields.get("previous_reading"),
            # "current_read_date":    normalise_date(
            #                             fields.get("current_read_date")
            #                             or fields.get("current_meter_read_date")
            #                         ),
            # "present_reading":      fields.get("present_reading"),
            "usage_quantity":       usage_quantity,
            "usage_unit":           usage_unit,
            # "previous_balance":     parse_amount(
            #                             fields.get("previous_balance")
            #                             or fields.get("previous_bill_amount")
            #                         ),
            # "payments_applied":     parse_amount(
            #                             fields.get("payments_applied")
            #                             or fields.get("payments")
            #                         ),
            # "current_charges":      parse_amount(
            #                             fields.get("current_charges")
            #                             or fields.get("new_charges")
            #                         ),
            # "taxes_and_fees":       parse_amount(
            #                             fields.get("taxes_and_fees")
            #                             or fields.get("sales_tax")
            #                         ),
            # "adjustments":          parse_amount(fields.get("adjustments")),
            "amount_due":           parse_amount(fields.get("amount_due")),
            "source_file":          result.get("source_file"),
            "extraction_date":      datetime.now().strftime("%Y-%m-%d"),
            # "extraction_rate":      result.get("extraction_rate"),
            # "needs_review":         ", ".join(
            #                             result.get("low_confidence_fields", [])
            #                         ) or None,
            "model_used":           result.get("model_used"),
        }

        cols         = ", ".join(row.keys())
        placeholders = ", ".join("?" * len(row))
        cur          = conn.cursor()
        cur.execute(
            f"INSERT OR IGNORE INTO bills ({cols}) VALUES ({placeholders})",
            list(row.values())
        )
        conn.commit()

        if cur.rowcount == 0:
            # Database-level UNIQUE constraint blocked the insert — same
            # outcome as the explicit check above, but catches any race
            # condition where two extractions of the same bill ran in
            # parallel and both passed the explicit check
            print(f"  ⏭ Skipped by DB constraint — "
                  f"account={account_number}, date={bill_date}")    # ← fixed
            return False

        print(f"  ✓ Saved bill_id={cur.lastrowid}  "
              f"customer_id={customer_id}  "
              f"account={account_number}  amount_due={row['amount_due']}")
        return True

    except Exception as e:
        print(f"  ✗ Database error: {e}")
        conn.rollback()
        return False

    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Customer lookup helpers — used by app.py for analytics and search
# ─────────────────────────────────────────────────────────────────────────────

def get_customer_summary(db_path: str = "troybanks_bills.db") -> list[dict]:
    """
    Returns a list of customers with their bill counts and totals.
    Useful for the Database tab to show a customers view.
    """
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        cur = conn.execute("""
            SELECT c.customer_id,
                   c.customer_name,
                   c.account_number,
                   COUNT(b.bill_id)            AS bill_count,
                   ROUND(SUM(b.amount_due), 2) AS total_spent,
                   MIN(b.bill_date)            AS first_bill,
                   MAX(b.bill_date)            AS latest_bill
            FROM   customers c
            LEFT JOIN bills b ON b.customer_id = c.customer_id
            GROUP BY c.customer_id
            ORDER BY total_spent DESC NULLS LAST
        """)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


def get_bills_for_customer(customer_id: int,
                             db_path: str = "troybanks_bills.db") -> list[dict]:
    """
    Returns all bills for a specific customer, ordered newest first.
    Used by the Database tab when the auditor clicks a customer.
    """
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        cur = conn.execute("""
            SELECT b.bill_id,
                   b.bill_date,
                   COALESCE(p.provider_name, 'Unknown') AS provider,
                   b.account_number,
                   b.amount_due,
                   b.usage_quantity,
                   b.usage_unit,
                   b.source_file
            FROM   bills b
            LEFT JOIN providers p ON b.provider_id = p.provider_id
            WHERE  b.customer_id = ?
            ORDER BY b.bill_date DESC
        """, (customer_id,))
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────────────────────────────────────

def process_batch(pdf_files: list,
                   db_path:   str = "troybanks_bills.db",
                   model:     str = "gemini-2.5-flash") -> dict:
    """
    Processes a list of bill files through Gemini and saves each to the db.
    Returns a summary dict with total/saved/skipped/failed counts.
    """
    from extractor import extract_bill
    import time

    summary = {"total": len(pdf_files), "saved": 0, "skipped": 0, "failed": 0}

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] {Path(pdf_file).name}")

        try:
            result = extract_bill(pdf_file, model=model)
            saved  = save_bill_to_db(result, db_path=db_path)
            if saved:
                summary["saved"]   += 1
            else:
                summary["skipped"] += 1

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            summary["failed"] += 1

        if i < len(pdf_files):
            time.sleep(4)

    print(f"\n{'='*45}")
    print(f"  Done — {summary['saved']} saved, "
          f"{summary['skipped']} skipped, "
          f"{summary['failed']} failed")
    return summary
