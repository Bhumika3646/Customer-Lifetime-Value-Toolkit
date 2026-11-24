"""Central configuration. Every tunable number in the project lives here."""
from pathlib import Path
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PACKAGE_ROOT / "raw_data"
RESULTS_DIR = PACKAGE_ROOT / "clv_results"
FIG_DIR = RESULTS_DIR / "figures"

RANDOM_STATE = 42

# ---- Dataset schema -------------------------------------------------------
# The Kaggle CSVs ship WITHOUT headers, so names are assigned here.
# NOTE: verify these against the dataset's Kaggle page before publication.
SCHEMAS = {
    "customers": ("customers.csv",
                  ["customer_id", "signup_date", "company_size", "industry",
                   "acquisition_channel", "country"]),
    "subscriptions": ("subscriptions.csv",
                      ["subscription_id", "customer_id", "start_date", "end_date",
                       "status", "plan_id"]),
    "invoices": ("invoices.csv",
                 ["invoice_id", "subscription_id", "invoice_date", "due_date",
                  "amount", "status"]),
    "payments": ("payments.csv",
                 ["payment_id", "subscription_id", "payment_date", "amount",
                  "status", "extra_blank"]),
    "plans": ("subscription_plans.csv",
              ["plan_id", "plan_name", "billing_period", "price"]),
    "churn": ("Churn_events.csv",
              ["event_id", "customer_id", "churn_date", "reason"]),
    "changes": ("plan_chnages.csv",   # (sic) file name typo is in the source data
                ["change_id", "subscription_id", "old_plan_id", "new_plan_id",
                 "change_date"]),
}
DATE_COLUMNS = {
    "customers": ["signup_date"],
    "subscriptions": ["start_date", "end_date"],
    "invoices": ["invoice_date", "due_date"],
    "payments": ["payment_date"],
    "churn": ["churn_date"],
    "changes": ["change_date"],
}

# ---- Time conventions -----------------------------------------------------
OBSERVATION_END = pd.Timestamp("2025-12-31")   # last date any record can carry
PAYMENT_DATA_END = pd.Timestamp("2025-11-30")  # last payment row in the data
DAYS_PER_MONTH = 365.25 / 12
DAYS_PER_YEAR = 365.25
BILLING_PERIOD_DAYS = {"monthly": DAYS_PER_MONTH, "annual": DAYS_PER_YEAR}

# ---- Survival models ------------------------------------------------------
MIN_SEGMENT_SIZE = 50          # smaller segments fall back to the global curve
TAIL_WINDOW_DAYS = 180         # window used to estimate the exponential tail hazard
TAIL_MIN_EVENTS = 15           # minimum events in window, else use overall hazard
COX_PENALIZER = 0.01

# ---- CLV ------------------------------------------------------------------
CLV_HORIZON_MONTHS = 36
ANNUAL_DISCOUNT_RATE = 0.10

# ---- Validation -----------------------------------------------------------
# Forecast origins. Each one fits models using ONLY information up to that date.
BACKTEST_CUTOFFS = [
    "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30",
    "2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30",
]
FIRST_TEST_CUTOFF = "2023-12-31"   # earlier cutoffs are used for training only
HORIZONS_MONTHS = [3, 6, 12]

# Fixed-origin customer-disjoint diagnostic. Historical training labels must be
# fully observed by this cutoff; held-out customer IDs are removed from ALL
# historical training snapshots before any supervised model is fit.
CUSTOMER_GROUP_HOLDOUT_CUTOFF = "2025-06-30"
CUSTOMER_GROUP_HOLDOUT_FRACTION = 0.20
