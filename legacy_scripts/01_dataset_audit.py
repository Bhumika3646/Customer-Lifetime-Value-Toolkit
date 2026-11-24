"""
Step 1: Dataset audit for the Customer Lifetime Value + Churn project.

Place this script in the same folder as "archive (8).zip", then run:
    python 01_dataset_audit.py

The CSVs in this ZIP are headerless, so this script supplies column names explicitly.
It does NOT train any models or overwrite the original ZIP.
"""

from pathlib import Path
import zipfile
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "archive (8).zip"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Column names inferred from the values and table relationships.
# Validate these against the Kaggle dataset documentation before publication.
SCHEMAS = {
    "customers.csv": [
        "customer_id", "signup_date", "company_size",
        "industry", "acquisition_channel", "country"
    ],
    "subscriptions.csv": [
        "subscription_id", "customer_id", "start_date",
        "end_date", "status", "plan_id"
    ],
    "invoices.csv": [
        "invoice_id", "subscription_id", "invoice_date",
        "due_date", "amount", "status"
    ],
    "payments.csv": [
        "payment_id", "subscription_id", "payment_date",
        "amount", "status", "extra_blank"
    ],
    "subscription_plans.csv": [
        "plan_id", "plan_name", "billing_period", "price"
    ],
    "Churn_events.csv": [
        "event_id", "customer_id", "churn_date", "reason"
    ],
    "plan_chnages.csv": [
        "change_id", "subscription_id", "old_plan_id",
        "new_plan_id", "change_date"
    ],
}

DATE_COLUMNS = {
    "customers.csv": ["signup_date"],
    "subscriptions.csv": ["start_date", "end_date"],
    "invoices.csv": ["invoice_date", "due_date"],
    "payments.csv": ["payment_date"],
    "Churn_events.csv": ["churn_date"],
    "plan_chnages.csv": ["change_date"],
}

def load_tables():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {ZIP_PATH.name}. Put this script next to the ZIP."
        )

    tables = {}
    with zipfile.ZipFile(ZIP_PATH) as archive:
        for filename, columns in SCHEMAS.items():
            if filename not in archive.namelist():
                raise FileNotFoundError(f"{filename} is missing from the ZIP.")
            df = pd.read_csv(archive.open(filename), header=None, names=columns)
            for column in DATE_COLUMNS.get(filename, []):
                df[column] = pd.to_datetime(df[column], errors="coerce")
            tables[filename] = df
    return tables

def main():
    tables = load_tables()
    lines = []
    lines.append("CUSTOMER LIFETIME VALUE PROJECT — DATASET AUDIT")
    lines.append("=" * 55)
    lines.append(f"ZIP file: {ZIP_PATH.name}")
    lines.append("Important: CSV column names were assigned by this script because the files are headerless.")
    lines.append("Confirm inferred column names with the original Kaggle documentation before publication.")
    lines.append("")

    for filename, df in tables.items():
        lines.append(f"[{filename}]")
        lines.append(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
        lines.append(f"Columns: {', '.join(df.columns)}")
        lines.append(f"Duplicate full rows: {int(df.duplicated().sum()):,}")
        lines.append("Missing values: " + str({
            col: int(count) for col, count in df.isna().sum().items() if count > 0
        }))
        for col in DATE_COLUMNS.get(filename, []):
            values = df[col].dropna()
            if not values.empty:
                lines.append(f"{col} range: {values.min().date()} to {values.max().date()}")
        for col in df.columns:
            if df[col].nunique(dropna=True) <= 12 and col not in DATE_COLUMNS.get(filename, []):
                lines.append(f"{col} value counts: {df[col].value_counts(dropna=False).to_dict()}")
        lines.append("Sample rows:")
        lines.append(df.head(3).to_string(index=False))
        lines.append("")

    # Relationship checks. These are diagnostics, not proof of data correctness.
    customers = tables["customers.csv"]
    subs = tables["subscriptions.csv"]
    invoices = tables["invoices.csv"]
    payments = tables["payments.csv"]
    plans = tables["subscription_plans.csv"]
    churn = tables["Churn_events.csv"]
    changes = tables["plan_chnages.csv"]

    lines.append("[RELATIONSHIP / CONSISTENCY CHECKS]")
    checks = [
        ("Subscription customer IDs found in customers",
         int(subs["customer_id"].isin(customers["customer_id"]).sum()), len(subs)),
        ("Subscription plan IDs found in plans",
         int(subs["plan_id"].isin(plans["plan_id"]).sum()), len(subs)),
        ("Invoice subscription IDs found in subscriptions",
         int(invoices["subscription_id"].isin(subs["subscription_id"]).sum()), len(invoices)),
        ("Payment subscription IDs found in subscriptions",
         int(payments["subscription_id"].isin(subs["subscription_id"]).sum()), len(payments)),
        ("Churn-event customer IDs found in customers",
         int(churn["customer_id"].isin(customers["customer_id"]).sum()), len(churn)),
        ("Plan-change subscription IDs found in subscriptions",
         int(changes["subscription_id"].isin(subs["subscription_id"]).sum()), len(changes)),
    ]
    for label, found, total in checks:
        lines.append(f"{label}: {found:,}/{total:,}")

    lines.append(f"Unique customers: {customers['customer_id'].nunique():,}")
    lines.append(f"Unique subscriptions: {subs['subscription_id'].nunique():,}")
    lines.append(f"Subscription ID equals customer ID for all rows: {bool((subs['subscription_id'] == subs['customer_id']).all())}")
    lines.append(f"Subscription status counts: {subs['status'].value_counts().to_dict()}")
    lines.append(f"Invoice status counts: {invoices['status'].value_counts().to_dict()}")
    lines.append(f"Payment status counts: {payments['status'].value_counts().to_dict()}")
    lines.append(f"Churn event count: {len(churn):,}")
    lines.append(f"Churned subscriptions: {int((subs['status'] == 'churned').sum()):,}")
    lines.append(f"Churned subscriptions with a matching churn event: {int(((subs['status'] == 'churned') & subs['customer_id'].isin(churn['customer_id'])).sum()):,}")
    lines.append(f"Invoice amount equals payment amount row-by-row: {bool((invoices['amount'].round(2).to_numpy() == payments['amount'].round(2).to_numpy()).all())}")
    lines.append(f"Invoice ID equals payment ID row-by-row: {bool((invoices['invoice_id'].to_numpy() == payments['payment_id'].to_numpy()).all())}")
    lines.append(f"Invoice subscription ID equals payment subscription ID row-by-row: {bool((invoices['subscription_id'].to_numpy() == payments['subscription_id'].to_numpy()).all())}")
    lines.append(f"All invoice statuses: {invoices['status'].unique().tolist()}")
    lines.append("")

    lines.append("[WARNINGS TO RESOLVE BEFORE MODELING]")
    lines.append("1. All invoice rows are marked paid, but some corresponding payment rows are failed. Investigate this inconsistency.")
    lines.append("2. Subscription status and churn-event details appear to directly encode the churn outcome. Do not use status, end_date, churn_date, or churn reason as pre-churn predictors.")
    lines.append("3. If the project claims future CLV prediction, define a prediction cutoff and future target window; use only data available before that cutoff as features.")
    lines.append("4. The dataset appears synthetic based on the identified Kaggle source; state this limitation clearly in the report.")
    lines.append("5. The column names above are inferred from the data layout and should be checked against the dataset documentation.")

    report = "\n".join(lines)
    report_path = OUTPUT_DIR / "dataset_audit_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved audit report to: {report_path}")

if __name__ == "__main__":
    main()
