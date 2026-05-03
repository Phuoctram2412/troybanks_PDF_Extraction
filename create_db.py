import sqlite3


def create_database(db_path: str = "troybanks_bills.db"):
    """
    Creates the database tables. Run once before first use.

    All tables start empty — rows are added automatically as bills are
    processed:
      - providers: registered when Gemini extracts a new provider_name
      - customers: registered when Gemini extracts a new customer_name
      - bills:     one row per processed bill, linked to both above

    To reset the database completely: delete troybanks_bills.db, then run
    this script again.
    """
    conn = sqlite3.connect(db_path, timeout=10)
    cur  = conn.cursor()

    cur.execute("PRAGMA foreign_keys = ON")

    # providers — populated automatically by Gemini extraction
    cur.execute("""
        CREATE TABLE IF NOT EXISTS providers (
            provider_id   INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            provider_name TEXT NOT NULL UNIQUE,
            bill_type     TEXT NOT NULL,
            created_at    TEXT DEFAULT (datetime('now'))
        )
    """)

    # customers — populated automatically as new customer_names appear
    # The UNIQUE (name, service_address) constraint allows the same person
    # to have multiple service addresses without forcing duplicate rows
    # (handled in code via lookup_or_create_customer with name-only fallback)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id     INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            customer_name            TEXT,
            account_number          TEXT
            created_at      TEXT DEFAULT (datetime('now')),
            UNIQUE (customer_name, account_number)
        )
    """)

    # bills — one row per bill, linked to a provider and a customer
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bills (
            bill_id              INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            provider_id          INTEGER REFERENCES providers(provider_id)
                                         ON DELETE SET NULL,
            customer_id          INTEGER REFERENCES customers(customer_id)
                                         ON DELETE SET NULL,
            customer_name        TEXT,
            account_number       TEXT,
            bill_date            TEXT,
            due_date             TEXT,
            meter_number         TEXT,
            usage_quantity       REAL,
            usage_unit           TEXT,
            amount_due           REAL,
            source_file          TEXT,
            extraction_date      TEXT DEFAULT (date('now')),
            model_used           TEXT,
            UNIQUE (account_number, bill_date)
        )
    """)

    # Indexes — speed up the most common queries
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bills_provider   ON bills(provider_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bills_customer   ON bills(customer_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bills_account    ON bills(account_number)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bills_bill_date  ON bills(bill_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bills_due_date   ON bills(due_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bills_amount_due ON bills(amount_due)")

    # Customer name lookup is the most-frequent customer query —
    # backs the lookup_or_create_customer name-only match path
    cur.execute("CREATE INDEX IF NOT EXISTS idx_customers_name   ON customers(customer_name)")

    conn.commit()
    conn.close()
    print(f"✓ Database ready → {db_path}")
    print("  providers table is empty — populated as bills are processed")
    print("  customers table is empty — populated as bills are processed")


def verify_database(db_path: str = "troybanks_bills.db"):
    """Prints a summary of what's in the database. Useful sanity check."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    print("─" * 50)

    # Providers
    cur.execute("SELECT provider_id, provider_name, bill_type FROM providers")
    providers = cur.fetchall()
    print(f"providers: {len(providers)} row(s)")
    for row in providers:
        print(f"  [{row[0]}] {row[1]:<30} {row[2]:<10} {row[3] or '—'}")

    # Customers
    cur.execute("SELECT customer_id, customer_name FROM customers")
    customers = cur.fetchall()
    print(f"customers: {len(customers)} row(s)")
    for row in customers:
        addr = (row[2] or '—')[:40]
        print(f"  [{row[0]}] {row[1]:<30} {addr}")

    # Bills count
    count = cur.execute("SELECT COUNT(*) FROM bills").fetchone()[0]
    print(f"bills:     {count} row(s)")

    conn.close()


if __name__ == "__main__":
    create_database()
    verify_database()