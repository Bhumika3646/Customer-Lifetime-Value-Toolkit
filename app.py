
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
CLV_RESULTS_DIR = BASE_DIR / "clv_results"

st.set_page_config(
    page_title="SubscriptIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL DASHBOARD STYLING
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 1.5rem 1.7rem;
            border-radius: 16px;
            background: linear-gradient(120deg, #101D3A, #1D4ED8);
            color: white;
            margin-bottom: 1.3rem;
        }

        .hero h1 {
            color: white;
            margin-bottom: .3rem;
        }

        .hero p {
            color: #DCE8FF;
            margin-bottom: 0;
        }

        div[data-testid="stMetric"] {
            background: #F6F8FC;
            border: 1px solid #E3E8F2;
            padding: 14px;
            border-radius: 12px;
        }

        .note {
            background: #FFF8E7;
            border-left: 4px solid #E9A23B;
            padding: 12px 15px;
            border-radius: 7px;
            color: #493A16;
            margin: 10px 0 18px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(filename):
    """Load a CSV from the project's outputs directory."""
    path = OUTPUT_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        st.warning(f"Could not load {filename}: {exc}")
        return pd.DataFrame()


@st.cache_data
def load_clv_csv(filename):
    """Load an output from the separate survival-based CLV pipeline."""
    path = CLV_RESULTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        st.warning(f"Could not load CLV result {filename}: {exc}")
        return pd.DataFrame()


@st.cache_data
def load_clv_json(filename):
    path = CLV_RESULTS_DIR / filename
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"Could not load CLV result {filename}: {exc}")
        return {}


snapshot = load_csv("customer_snapshot_2025-06-30.csv")
churn_predictions = load_csv("best_churn_test_predictions.csv")
revenue_predictions = load_csv("best_regression_test_predictions.csv")
plan_predictions = load_csv("plan_wise_predictions.csv")
rolling_results = load_csv("rolling_temporal_validation_results.csv")
rolling_predictions = load_csv("rolling_temporal_predictions.csv")
feature_importance = load_csv("xgboost_feature_importance.csv")
baseline_comparison = load_csv("revenue_baseline_comparison.csv")
plan_errors = load_csv("plan_wise_error_analysis.csv")
prediction_bias = load_csv("prediction_bias_analysis.csv")
range_errors = load_csv("revenue_range_error_analysis.csv")
monthly_results = load_csv("monthly_only_regression_results.csv")
monthly_validation = load_csv("monthly_only_regression_validation_results.csv")
churn_results = load_csv("temporal_churn_results.csv")
regression_results = load_csv("temporal_regression_results.csv")
hybrid_results = load_csv("hybrid_test_results.csv")

# Separate survival-based CLV outputs (36-month horizon at 2025-12-31).
clv_customers = load_clv_csv("03_customer_clv.csv")
clv_segments = load_clv_csv("03_clv_by_segment.csv")
clv_survival = load_clv_csv("02_survival_by_segment.csv")
clv_sensitivity = load_clv_csv("03_clv_sensitivity.csv")
clv_unbilled_sensitivity = load_clv_csv("03_unbilled_billing_sensitivity.csv")
clv_revenue_backtest = load_clv_csv("04_revenue_backtest_summary.csv")
clv_retention_backtest = load_clv_csv("05_retention_backtest_summary.csv")
clv_customer_grouped_holdout = load_clv_csv("06_customer_grouped_holdout.csv")
clv_unbilled_breakdown = load_clv_csv("03_unbilled_breakdown.csv")
clv_grouped_by_plan = load_clv_csv("06_grouped_revenue_by_plan.csv")
clv_unc_total = load_clv_csv("09_bootstrap_clv_total.csv")
clv_unc_segments = load_clv_csv("09_bootstrap_clv_by_segment.csv")
clv_unc_reps = load_clv_csv("09_bootstrap_total_replicates.csv")
clv_unc_draws = load_clv_csv("09_predictive_draws.csv")
clv_interval_coverage = load_clv_csv("04_revenue_interval_coverage.csv")
clv_scenarios = load_clv_csv("09_scenario_grid.csv")
clv_value_risk_summary = load_clv_csv("07_value_risk_segment_summary.csv")
clv_retention_recommendations = load_clv_csv("07_customer_retention_recommendations.csv")
clv_shap_global = load_clv_csv("08_shap_global_summary.csv")
clv_shap_local = load_clv_csv("08_shap_local_examples.csv")
clv_audit = load_clv_json("01_data_audit.json")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def money(value):
    if pd.isna(value):
        return "—"
    return f"₹{value:,.2f}"


def number(value):
    if pd.isna(value):
        return "—"
    return f"{value:,.0f}"


def pct(value):
    if pd.isna(value):
        return "—"
    return f"{value:.1f}%"


def first_existing(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


# ============================================================
# HEADER AND NAVIGATION
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>SubscriptIQ</h1>
        <p>90-Day Revenue Prediction · Churn Prediction · Survival-Based CLV</p>
        <p>Customer Lifetime Value Research Project | Snapshot: 30 June 2025</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("SubscriptIQ")
st.sidebar.caption("Customer Intelligence Platform")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Executive Overview",
        "Customer Segments",
        "Revenue Forecast",
        "Churn & Retention",
        "Model Evaluation",
        "Survival-Based CLV",
        "Retention Recommendations",
        "Uncertainty & Scenarios",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Research prototype · Historical dataset")


# ============================================================
# COMMON DATA PREPARATION
# ============================================================

if not snapshot.empty:
    numeric_columns = [
        "price",
        "revenue_30d",
        "revenue_60d",
        "revenue_90d",
        "historical_successful_revenue",
        "historical_successful_payment_count",
        "successful_payments_30d",
        "successful_payments_60d",
        "successful_payments_90d",
        "target_future_90d_revenue",
        "target_churn_within_90d",
    ]

    for col in numeric_columns:
        if col in snapshot.columns:
            snapshot[col] = pd.to_numeric(
                snapshot[col], errors="coerce"
            )

    if "signup_date" in snapshot.columns:
        snapshot["signup_date"] = pd.to_datetime(
            snapshot["signup_date"], errors="coerce"
        )

    if "plan_name" not in snapshot.columns:
        snapshot["plan_name"] = "Unknown"

    if "billing_period" not in snapshot.columns:
        snapshot["billing_period"] = "Unknown"

    snapshot["plan_name"] = (
        snapshot["plan_name"].fillna("Unknown").astype(str)
    )

    snapshot["billing_period"] = (
        snapshot["billing_period"].fillna("Unknown").astype(str)
    )

    if "target_churn_within_90d" in snapshot.columns:
        snapshot["churn_label"] = snapshot["target_churn_within_90d"]
    else:
        snapshot["churn_label"] = np.nan


# ============================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.subheader("Executive Overview")

    st.write(
        "This research prototype analyzes subscription customers using two "
        "main prediction tasks: future 90-day revenue prediction and churn "
        "prediction. The results support customer segmentation and retention "
        "analysis, subject to the limitations of the available historical data."
    )

    if snapshot.empty:
        st.error(
            "Customer snapshot is missing. Check the outputs folder."
        )
        st.stop()

    total_customers = (
        snapshot["customer_id"].nunique()
        if "customer_id" in snapshot.columns
        else len(snapshot)
    )

    total_subscriptions = (
        snapshot["subscription_id"].nunique()
        if "subscription_id" in snapshot.columns
        else len(snapshot)
    )

    historical_revenue = (
        snapshot["historical_successful_revenue"].sum()
        if "historical_successful_revenue" in snapshot.columns
        else np.nan
    )

    target_revenue = (
        snapshot["target_future_90d_revenue"].sum()
        if "target_future_90d_revenue" in snapshot.columns
        else np.nan
    )

    churn_rate = (
        snapshot["churn_label"].mean() * 100
        if snapshot["churn_label"].notna().any()
        else np.nan
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Customers", number(total_customers))
    c2.metric("Subscriptions", number(total_subscriptions))
    c3.metric(
        "Historical successful revenue",
        money(historical_revenue),
    )
    c4.metric(
        "Recorded 90-day churn rate",
        pct(churn_rate),
    )

    st.markdown(
        '<div class="note"><b>Interpretation:</b> Revenue and churn figures '
        'below are dataset-derived indicators. Future revenue means '
        'successful-payment revenue recorded in the 90-day target window; '
        'it is not lifetime revenue or profit.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        if "plan_name" in snapshot.columns:
            plan_counts = (
                snapshot.groupby("plan_name", dropna=False)
                .size()
                .reset_index(name="Customers")
                .sort_values("Customers", ascending=False)
            )

            fig = px.bar(
                plan_counts,
                x="plan_name",
                y="Customers",
                color="plan_name",
                title="Customers by Subscription Plan",
                labels={"plan_name": "Plan"},
            )

            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

    with right:
        if "acquisition_channel" in snapshot.columns:
            channel_counts = (
                snapshot["acquisition_channel"]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("Channel")
                .reset_index(name="Customers")
            )

            fig = px.pie(
                channel_counts,
                names="Channel",
                values="Customers",
                hole=0.55,
                title="Customer Acquisition Channels",
            )

            st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)

    with left:
        if "company_size" in snapshot.columns:
            size_counts = (
                snapshot["company_size"]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("Company Size")
                .reset_index(name="Customers")
            )

            fig = px.bar(
                size_counts,
                x="Company Size",
                y="Customers",
                title="Customer Distribution by Company Size",
            )

            st.plotly_chart(fig, width="stretch")

    with right:
        if "industry" in snapshot.columns:
            industry_counts = (
                snapshot["industry"]
                .fillna("Unknown")
                .value_counts()
                .head(12)
                .rename_axis("Industry")
                .reset_index(name="Customers")
            )

            fig = px.bar(
                industry_counts.sort_values("Customers"),
                x="Customers",
                y="Industry",
                orientation="h",
                title="Top Customer Industries",
            )

            st.plotly_chart(fig, width="stretch")

    st.subheader("Research and data-quality notes")

    st.warning(
        "Annual-plan invoice/payment records were absent from the 2025 data "
        "audit. Revenue targets for annual subscribers may therefore be "
        "under-recorded. Treat revenue model results as provisional."
    )

    st.info(
        "The current snapshot is dated 30 June 2025. The dashboard describes "
        "the available dataset, not a live billing system."
    )


# ============================================================
# PAGE 2: CUSTOMER SEGMENTS
# ============================================================

elif page == "Customer Segments":

    st.subheader("Customer Segments")

    st.write(
        "Explore how subscription plans and customer characteristics relate "
        "to historical payment activity and recorded future revenue."
    )

    if snapshot.empty:
        st.error("Customer snapshot is missing.")
        st.stop()

    plans = sorted(
        snapshot["plan_name"].dropna().unique().tolist()
    )

    selected_plans = st.multiselect(
        "Subscription plans",
        plans,
        default=plans,
    )

    filtered = snapshot[
        snapshot["plan_name"].isin(selected_plans)
    ].copy()

    if filtered.empty:
        st.info("No customers match the selected plans.")
        st.stop()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Customers in selection",
        number(filtered["customer_id"].nunique())
        if "customer_id" in filtered.columns
        else number(len(filtered)),
    )

    c2.metric(
        "Average historical revenue",
        money(filtered["historical_successful_revenue"].mean())
        if "historical_successful_revenue" in filtered.columns
        else "—",
    )

    c3.metric(
        "Average recorded future 90-day revenue",
        money(filtered["target_future_90d_revenue"].mean())
        if "target_future_90d_revenue" in filtered.columns
        else "—",
    )

    required_plan_columns = {
        "plan_name",
        "customer_id",
        "price",
        "historical_successful_revenue",
        "target_future_90d_revenue",
    }

    if required_plan_columns.issubset(filtered.columns):
        agg = (
            filtered.groupby("plan_name")
            .agg(
                Customers=("customer_id", "nunique"),
                Mean_Price=("price", "mean"),
                Mean_Historical_Revenue=(
                    "historical_successful_revenue", "mean"
                ),
                Mean_Future_90d_Revenue=(
                    "target_future_90d_revenue", "mean"
                ),
            )
            .reset_index()
        )

        st.plotly_chart(
            px.bar(
                agg,
                x="plan_name",
                y=[
                    "Mean_Historical_Revenue",
                    "Mean_Future_90d_Revenue",
                ],
                barmode="group",
                title="Average Revenue Indicators by Plan",
                labels={
                    "value": "Revenue",
                    "plan_name": "Plan",
                    "variable": "Measure",
                },
            ),
            width="stretch",
        )

        st.dataframe(
            agg,
            width="stretch",
            hide_index=True,
        )

    left, right = st.columns(2)

    with left:
        if {
            "billing_period",
            "target_future_90d_revenue",
        }.issubset(filtered.columns):

            fig = px.box(
                filtered,
                x="billing_period",
                y="target_future_90d_revenue",
                points=False,
                title="Recorded Future Revenue by Billing Period",
                labels={
                    "billing_period": "Billing Period",
                    "target_future_90d_revenue": (
                        "Recorded 90-day revenue"
                    ),
                },
            )

            st.plotly_chart(fig, width="stretch")

    with right:
        if {
            "company_size",
            "target_future_90d_revenue",
        }.issubset(filtered.columns):

            company_agg = (
                filtered.groupby("company_size", dropna=False)
                .agg(
                    Mean_Revenue=("target_future_90d_revenue", "mean"),
                    Customers=("customer_id", "nunique"),
                )
                .reset_index()
            )

            fig = px.bar(
                company_agg,
                x="company_size",
                y="Mean_Revenue",
                title="Average Recorded Future Revenue by Company Size",
                labels={
                    "company_size": "Company size",
                    "Mean_Revenue": "Mean revenue",
                },
            )

            st.plotly_chart(fig, width="stretch")

    st.subheader("Customer records")

    display_cols = [
        c
        for c in [
            "customer_id",
            "plan_name",
            "billing_period",
            "price",
            "company_size",
            "industry",
            "acquisition_channel",
            "country",
            "historical_successful_revenue",
            "revenue_90d",
            "target_future_90d_revenue",
            "target_churn_within_90d",
        ]
        if c in filtered.columns
    ]

    st.dataframe(
        filtered[display_cols].head(1000),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Showing up to 1,000 matching rows. IDs are dataset identifiers."
    )


# ============================================================
# PAGE 3: REVENUE FORECAST
# ============================================================

elif page == "Revenue Forecast":

    st.subheader("90-Day Future Revenue Prediction — Monthly-Plan Test Cohort")

    st.write(
        "Evaluate saved model predictions against recorded successful-payment "
        "revenue during the 90-day target window. This is a short-term revenue "
        "proxy, not a prediction of total customer lifetime value or profit."
    )

    st.caption(
        "Evaluation scope: monthly-plan customers in the temporal test snapshot "
        "at 2025-06-30, with the target window ending 2025-09-28. Annual-plan "
        "customers are excluded from this sensitivity analysis because the "
        "audited 2025 data lacks annual-plan payment/invoice records. "
        "The forecast metrics below use the saved predictions for this "
        "monthly-only cohort; the separate XGBoost model in Model Evaluation "
        "was trained with a different pipeline and can report different scores."
    )

    if not plan_predictions.empty:

        df = plan_predictions.copy()

        actual_col = first_existing(
            df,
            ["actual_future_90d_revenue"],
        )

        pred_col = first_existing(
            df,
            ["full_xgboost_prediction"],
        )

        plan_col = first_existing(
            df,
            ["plan_name"],
        )

        if actual_col and pred_col:

            df[actual_col] = pd.to_numeric(
                df[actual_col], errors="coerce"
            )

            df[pred_col] = pd.to_numeric(
                df[pred_col], errors="coerce"
            )

            df = df.dropna(
                subset=[actual_col, pred_col]
            )

            if plan_col:
                available = sorted(
                    df[plan_col].fillna("Unknown").unique()
                )

                selected = st.multiselect(
                    "Plans for prediction analysis",
                    available,
                    default=available,
                )

                df = df[
                    df[plan_col]
                    .fillna("Unknown")
                    .isin(selected)
                ]

            if not df.empty:

                df["Absolute Error"] = (
                    df[actual_col] - df[pred_col]
                ).abs()

                mae = df["Absolute Error"].mean()

                rmse = np.sqrt(
                    np.mean(
                        (df[actual_col] - df[pred_col]) ** 2
                    )
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Rows evaluated",
                    number(len(df)),
                )

                c2.metric(
                    "Mean absolute error",
                    money(mae),
                )

                c3.metric(
                    "Root mean squared error",
                    money(rmse),
                )

                scatter = px.scatter(
                    df,
                    x=actual_col,
                    y=pred_col,
                    color=plan_col if plan_col else None,
                    opacity=0.55,
                    title="Actual vs Predicted Future 90-Day Revenue",
                    labels={
                        actual_col: "Actual recorded revenue",
                        pred_col: "Model prediction",
                    },
                    hover_data=[plan_col] if plan_col else None,
                )

                max_val = max(
                    float(df[actual_col].max()),
                    float(df[pred_col].max()),
                    1,
                )

                scatter.add_trace(
                    go.Scatter(
                        x=[0, max_val],
                        y=[0, max_val],
                        mode="lines",
                        name="Perfect prediction",
                        line={
                            "dash": "dash",
                            "color": "gray",
                        },
                    )
                )

                st.plotly_chart(
                    scatter,
                    width="stretch",
                )

                if plan_col:

                    plan_agg = (
                        df.groupby(plan_col)
                        .agg(
                            Customers=(actual_col, "size"),
                            Actual_Mean=(actual_col, "mean"),
                            Predicted_Mean=(pred_col, "mean"),
                        )
                        .reset_index()
                    )

                    plan_agg["Mean Bias"] = (
                        plan_agg["Predicted_Mean"]
                        - plan_agg["Actual_Mean"]
                    )

                    st.plotly_chart(
                        px.bar(
                            plan_agg,
                            x=plan_col,
                            y=[
                                "Actual_Mean",
                                "Predicted_Mean",
                            ],
                            barmode="group",
                            title=(
                                "Actual vs Predicted Average Revenue by Plan"
                            ),
                        ),
                        width="stretch",
                    )

                    st.dataframe(
                        plan_agg,
                        width="stretch",
                        hide_index=True,
                    )

                st.subheader("Prediction records")

                st.dataframe(
                    df.head(1000),
                    width="stretch",
                    hide_index=True,
                )

                st.download_button(
                    "Download filtered predictions",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="filtered_revenue_predictions.csv",
                    mime="text/csv",
                )

            else:
                st.info(
                    "No prediction records match the filters."
                )

        else:
            st.warning(
                "The plan-wise prediction file has unexpected columns."
            )

    else:
        st.warning(
            "Plan-wise prediction file is not available."
        )

    st.markdown(
        '<div class="note"><b>Important:</b> This page evaluates saved '
        'predictions against recorded successful-payment revenue in the next '
        '90 days. It does not measure total customer lifetime value or profit. '
        'It evaluates saved predictions only and is not a production scoring '
        'service.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 4: CHURN AND RETENTION
# ============================================================

elif page == "Churn & Retention":

    st.subheader("Churn & Retention Intelligence")

    st.write(
        "Review recorded 90-day churn outcomes and the saved churn model's "
        "test predictions."
    )

    if not churn_predictions.empty:

        df = churn_predictions.copy()

        if {
            "actual_churn",
            "predicted_churn_probability",
        }.issubset(df.columns):

            df["actual_churn"] = pd.to_numeric(
                df["actual_churn"], errors="coerce"
            )

            df["predicted_churn_probability"] = pd.to_numeric(
                df["predicted_churn_probability"],
                errors="coerce",
            )

            df = df.dropna(
                subset=[
                    "actual_churn",
                    "predicted_churn_probability",
                ]
            )

            df["Actual outcome"] = df["actual_churn"].map(
                {
                    0: "No recorded churn",
                    1: "Recorded churn",
                }
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Evaluated customers",
                number(len(df)),
            )

            c2.metric(
                "Recorded churn rate",
                pct(100 * df["actual_churn"].mean()),
            )

            c3.metric(
                "Mean predicted churn probability",
                pct(
                    100
                    * df["predicted_churn_probability"].mean()
                ),
            )

            st.plotly_chart(
                px.histogram(
                    df,
                    x="predicted_churn_probability",
                    color="Actual outcome",
                    nbins=25,
                    barmode="overlay",
                    opacity=0.7,
                    title=(
                        "Predicted Churn Probability by Actual Outcome"
                    ),
                    labels={
                        "predicted_churn_probability": (
                            "Predicted probability"
                        )
                    },
                ),
                width="stretch",
            )

            threshold = st.slider(
                "Illustrative churn alert threshold",
                min_value=0.05,
                max_value=0.95,
                value=0.60,
                step=0.05,
                help=(
                    "Changing this threshold changes the displayed alert "
                    "group. It does not retrain the model."
                ),
            )

            df["Alert"] = (
                df["predicted_churn_probability"] >= threshold
            )

            alerts = df[df["Alert"]]

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Customers above threshold",
                number(len(alerts)),
            )

            c2.metric(
                "Recorded churn within alert group",
                pct(
                    100 * alerts["actual_churn"].mean()
                )
                if len(alerts)
                else "—",
            )

            total_churn = (
                df["actual_churn"] == 1
            ).sum()

            captured_churn = df.loc[
                df["actual_churn"] == 1,
                "Alert",
            ].sum()

            c3.metric(
                "Share of recorded churn captured",
                pct(
                    100 * captured_churn / max(total_churn, 1)
                ),
            )

            st.caption(
                "The threshold slider is for exploration. A real retention "
                "campaign needs validation of costs, benefits, calibration, "
                "and operational capacity before deployment."
            )

            st.subheader(
                "Highest predicted churn probabilities"
            )

            high_risk = (
                df.sort_values(
                    "predicted_churn_probability",
                    ascending=False,
                )
                .head(100)
            )

            st.dataframe(
                high_risk,
                width="stretch",
                hide_index=True,
            )

            st.download_button(
                "Download churn analysis",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="churn_analysis.csv",
                mime="text/csv",
            )

        else:
            st.warning(
                "Churn prediction columns are missing."
            )

    else:
        st.warning(
            "Saved churn predictions are unavailable."
        )

    st.markdown(
        '<div class="note"><b>Retention interpretation:</b> A high predicted '
        'probability is a prioritization signal, not proof that a customer '
        'will leave. The target represents recorded churn events in a 90-day '
        'window and may not capture all cancellations.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 5: MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.subheader("Model Evaluation and Reliability")

    st.write(
        "Compare available validation results and inspect how performance "
        "changes across temporal test windows. These results come from separate "
        "experiments, so check each section's evaluation scope before comparing scores."
    )

    if not rolling_results.empty:

        st.subheader("Rolling Temporal Validation — Expanding Training Windows")
        st.caption(
            "Each test cutoff evaluates the following 90-day successful-payment "
            "revenue target. Training uses only the earlier cutoffs listed in "
            "training_cutoffs. Customers can appear in more than one temporal "
            "snapshot, so this is time-aware validation but not customer-independent "
            "validation."
        )

        df = rolling_results.copy()

        for col in [
            "MAE",
            "RMSE",
            "R2",
            "test_rows",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col], errors="coerce"
                )

        if "model" in df.columns:

            available_metrics = [
                col
                for col in ["MAE", "RMSE", "R2"]
                if col in df.columns
            ]

            if available_metrics:

                metric = st.selectbox(
                    "Metric to compare",
                    available_metrics,
                    index=0,
                )

                if "test_cutoff" in df.columns:

                    hover_columns = [
                        col
                        for col in [
                            "test_rows",
                            "training_rows",
                            "target_horizon_end",
                        ]
                        if col in df.columns
                    ]

                    fig = px.line(
                        df,
                        x="test_cutoff",
                        y=metric,
                        color="model",
                        markers=True,
                        title=(
                            f"Rolling Temporal Validation: {metric}"
                        ),
                        hover_data=hover_columns,
                    )

                    st.plotly_chart(
                        fig,
                        width="stretch",
                    )

                st.dataframe(
                    df,
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "Metric guide: lower MAE and RMSE mean smaller prediction errors; "
                    "higher R² indicates more variation explained. A model can lead "
                    "on one metric without leading on all three."
                )
                st.caption(
                    f"This file contains {df['test_cutoff'].nunique()} test windows. "
                    "The dataset appears synthetic, so these scores do not establish "
                    "real-world performance."
                )

    else:
        st.info(
            "Rolling temporal validation results are unavailable."
        )

    left, right = st.columns(2)

    with left:

        if not baseline_comparison.empty:

            st.subheader(
                "Fixed Temporal Holdout: Model vs Baselines — Monthly Plans"
            )
            st.caption(
                "Separate experiment from rolling validation: trained on the "
                "2024-06-30 snapshot and evaluated on the 2025-06-30 snapshot, "
                "with the 90-day target ending 2025-09-28. Only monthly-plan "
                "customers are included. The XGBoost pipeline in this comparison "
                "is trained separately from the XGBoost predictions shown on the "
                "Revenue Forecast page, so its score can differ even though the "
                "test cohort and target horizon are the same. Do not combine this "
                "table with the rolling-window scores."
            )

            st.dataframe(
                baseline_comparison,
                width="stretch",
                hide_index=True,
            )

            metric_col = first_existing(
                baseline_comparison,
                ["MAE", "mae"],
            )

            model_col = first_existing(
                baseline_comparison,
                ["Model", "model"],
            )

            if metric_col and model_col:

                st.plotly_chart(
                    px.bar(
                        baseline_comparison,
                        x=model_col,
                        y=metric_col,
                        title="Fixed-Holdout MAE Comparison (lower is better)",
                    ),
                    width="stretch",
                )

    with right:

        if not feature_importance.empty:

            st.subheader("XGBoost Feature Importance")

            if {
                "original_feature",
                "importance_pct",
            }.issubset(feature_importance.columns):

                fi = feature_importance.copy()

                fi["importance_pct"] = pd.to_numeric(
                    fi["importance_pct"],
                    errors="coerce",
                )

                fi = (
                    fi.dropna(subset=["importance_pct"])
                    .sort_values("importance_pct")
                    .tail(15)
                )

                st.plotly_chart(
                    px.bar(
                        fi,
                        x="importance_pct",
                        y="original_feature",
                        orientation="h",
                        title="Top Features by Recorded Importance",
                        labels={
                            "importance_pct": "Importance (%)",
                            "original_feature": "Feature",
                        },
                    ),
                    width="stretch",
                )

                st.caption(
                    "Feature importance describes model reliance, not "
                    "causality or proof that a feature causes churn or revenue."
                )

    if not plan_errors.empty:

        st.subheader("Prediction Error by Subscription Plan")

        st.dataframe(
            plan_errors,
            width="stretch",
            hide_index=True,
        )

    if not prediction_bias.empty:

        st.subheader("Prediction Bias by Plan")

        st.dataframe(
            prediction_bias,
            width="stretch",
            hide_index=True,
        )

    if not range_errors.empty:

        st.subheader("Errors by Revenue Range")

        st.dataframe(
            range_errors,
            width="stretch",
            hide_index=True,
        )

    if not monthly_results.empty:

        st.subheader("Monthly-Only Sensitivity Test")

        st.caption(
            "This is a separate subset analysis and should not be compared "
            "directly with the full-population test as if both used the same "
            "customers."
        )

        st.dataframe(
            monthly_results,
            width="stretch",
            hide_index=True,
        )

    if not monthly_validation.empty:

        st.subheader("Monthly-Only Validation Results")

        st.dataframe(
            monthly_validation,
            width="stretch",
            hide_index=True,
        )

    st.subheader("Objective-Aligned Regression Model Comparison — Customer-Grouped Holdout")
    if not clv_customer_grouped_holdout.empty:
        grouped_df = clv_customer_grouped_holdout.copy()
        if "task" in grouped_df.columns:
            revenue_eval = grouped_df[grouped_df["task"] == "3-month revenue"].copy()
            if not revenue_eval.empty:
                st.caption("All four requested regressors use the same 3-month target, fixed cutoff and held-out customer cohort. This customer-grouped result is separate from the rolling-origin averages above.")
                display_cols = [c for c in ["model", "train_customer_groups", "test_customer_groups", "customer_overlap", "MAE", "RMSE", "R2", "portfolio_bias_pct"] if c in revenue_eval.columns]
                st.dataframe(revenue_eval[display_cols], width="stretch", hide_index=True)
                if {"model", "RMSE"}.issubset(revenue_eval.columns):
                    chart_df = revenue_eval.dropna(subset=["RMSE"]).sort_values("RMSE")
                    st.plotly_chart(px.bar(chart_df, x="model", y="RMSE", title="Customer-Grouped Revenue Holdout — RMSE (lower is better)"), width="stretch")
            if not clv_grouped_by_plan.empty:
                st.caption("Accuracy within each plan. The pooled R² above is inflated by price differences between plans; within a plan the ML models have R² near 0 and do not beat the plan-price baseline on MAE.")
                st.dataframe(clv_grouped_by_plan, width="stretch", hide_index=True)
            churn_eval = grouped_df[grouped_df["task"] == "3-month churn"].copy()
            if not churn_eval.empty:
                st.caption("Churn models are evaluated on the same unseen-customer cohort using ROC-AUC, PR-AUC and Brier score. For binary churn, lower Brier is better; ROC-AUC/PR-AUC are ranking metrics.")
                display_cols = [c for c in ["model", "train_customer_groups", "test_customer_groups", "customer_overlap", "ROC_AUC", "PR_AUC", "Brier", "calibration_gap"] if c in churn_eval.columns]
                st.dataframe(churn_eval[display_cols], width="stretch", hide_index=True)
    else:
        st.info("Customer-grouped model comparison is not available. Run `python run_pipeline.py` to regenerate the objective-aligned comparisons.")

    st.subheader("Temporal Behavioral Features")
    trend_features = pd.DataFrame([
        {"feature": "revenue_30d / revenue_60d / revenue_90d", "meaning": "Successful-payment revenue in rolling 30-, 60-, and 90-day windows."},
        {"feature": "revenue_prev30d", "meaning": "Revenue in the 31–60 day period before the cutoff."},
        {"feature": "revenue_days61_90", "meaning": "Revenue in the 61–90 day period before the cutoff."},
        {"feature": "revenue_delta_30d_vs_prev30d", "meaning": "Absolute change: most recent 30-day revenue minus previous 30-day revenue."},
        {"feature": "revenue_trend_ratio_30d_vs_prev30d", "meaning": "Bounded normalized revenue change; helps compare changes across customer sizes."},
        {"feature": "payment_count_delta_30d_vs_prev30d", "meaning": "Change in successful payment count versus the preceding 30 days."},
        {"feature": "failed_count_delta_30d_vs_prev30d", "meaning": "Change in failed-payment count versus the preceding 30 days."},
    ])
    st.dataframe(trend_features, width="stretch", hide_index=True)
    st.caption("The latest CLV pipeline regenerates these features only from records dated on or before each cutoff. The separate legacy forecast files shown above were generated by an earlier pipeline and should not be described as using every new delta feature.")

    if not clv_shap_global.empty:
        st.subheader("Explainable AI — SHAP Global Feature Impact")
        shap_df = clv_shap_global.copy()
        for col in ["mean_abs_shap", "mean_signed_shap"]:
            if col in shap_df.columns:
                shap_df[col] = pd.to_numeric(shap_df[col], errors="coerce")
        if "task" in shap_df.columns:
            shap_task = st.selectbox("SHAP explanation target", sorted(shap_df["task"].dropna().unique().tolist()), key="objective_shap_task")
            shap_view = shap_df[shap_df["task"] == shap_task].dropna(subset=["mean_abs_shap"]).sort_values("mean_abs_shap").tail(15)
            if not shap_view.empty:
                st.plotly_chart(px.bar(shap_view, x="mean_abs_shap", y="source_feature", orientation="h", title=f"Top SHAP Features — {shap_task}", labels={"mean_abs_shap": "Mean absolute SHAP value", "source_feature": "Original feature"}), width="stretch")
                st.dataframe(shap_view.sort_values("mean_abs_shap", ascending=False), width="stretch", hide_index=True)
        st.caption("SHAP explains the supervised XGBoost 90-day revenue proxy and 3-month churn classifier on held-out customers. Revenue SHAP is in target units; binary churn SHAP is in log-odds. SHAP attribution is not causality and is not a direct decomposition of the separate Cox survival CLV score.")
        if not clv_shap_local.empty:
            with st.expander("Local SHAP examples (sample customers; anonymized example numbers)"):
                st.dataframe(clv_shap_local, width="stretch", hide_index=True)
    else:
        st.warning("SHAP global output was not generated. Check clv_results/08_shap_status.txt after running the pipeline before claiming the explainable-AI objective is complete.")

    st.warning(
        "Known limitations: the dataset appears synthetic, so the reported "
        "scores do not establish real-world generalization. Annual-plan "
        "payment/invoice records were absent from the audited 2025 data; the "
        "monthly-only results are a sensitivity analysis, not an estimate for "
        "all customers. The target is recorded successful-payment revenue over "
        "90 days, not true lifetime value or profit. Customers overlap across "
        "some temporal snapshots, so rolling validation is not customer-independent. "
        "Treat results as exploratory."
    )


# ============================================================
# PAGE 6: SURVIVAL-BASED CUSTOMER LIFETIME VALUE
# ============================================================

elif page == "Survival-Based CLV":

    st.subheader("Survival-Based Customer Lifetime Value")
    st.write(
        "This section estimates expected discounted successful-payment revenue "
        "using customer retention probabilities. It adds a longer-horizon value "
        "estimate to the existing 90-day revenue and churn analyses."
    )
    st.caption(
        "CLV scoring snapshot: 31 December 2025 · 36-month forecast horizon · "
        "10% annual discount rate. This is a separate snapshot from the existing "
        "90-day Revenue Forecast test cohort dated 30 June 2025."
    )

    if clv_customers.empty or clv_segments.empty:
        st.warning(
            "Survival-based CLV outputs are not available. Install dependencies "
            "from requirements.txt, then run `python run_pipeline.py` from the "
            "project folder to generate them."
        )
    else:
        cc = clv_customers.copy()
        for col in ["clv_12m", "clv_36m", "retention_prob_12m", "expected_active_months_36m", "plan_price"]:
            if col in cc.columns:
                cc[col] = pd.to_numeric(cc[col], errors="coerce")
        cc["clv_12m"] = cc["clv_12m"].fillna(0)
        cc["clv_36m"] = cc["clv_36m"].fillna(0)
        positive = cc[cc["clv_36m"] > 0]
        total_clv = cc["clv_36m"].sum()
        avg_positive_clv = positive["clv_36m"].mean() if not positive.empty else np.nan
        mean_retention = cc["retention_prob_12m"].mean() if "retention_prob_12m" in cc.columns else np.nan

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active subscriptions scored", number(cc["subscription_id"].nunique() if "subscription_id" in cc else len(cc)))
        m2.metric("Estimated total 36-month CLV", money(total_clv))
        m3.metric("Average 36-month CLV · positive-value customers", money(avg_positive_clv))
        m4.metric("Mean 12-month retention probability", pct(mean_retention * 100 if pd.notna(mean_retention) else np.nan))

        st.markdown(
            '<div class="note"><b>Interpretation:</b> CLV here means estimated discounted '
            'successful-payment revenue over the selected horizon. It is not profit, '
            'does not include costs or acquisition spend, and is not an unlimited-lifetime estimate.</div>',
            unsafe_allow_html=True,
        )
        st.warning(
            "Annual-plan limitation: annual-plan segments are included in these CLV estimates, "
            "but annual renewal payments are missing from 2025 in this dataset. The revenue "
            "backtest therefore cannot validate annual-plan forecasts; annual-plan values and "
            "the portfolio total should be treated as provisional model-based projections."
        )

        left, right = st.columns(2)
        with left:
            positive_value_customers = cc.loc[cc["clv_36m"] > 0, ["clv_36m"]].dropna()
            if not positive_value_customers.empty:
                fig = px.histogram(
                    positive_value_customers, x="clv_36m", nbins=35,
                    title="Distribution of 36-Month CLV (positive-value customers)",
                    labels={"clv_36m": "Estimated discounted revenue (₹)"},
                )
                st.plotly_chart(fig, width="stretch")
        with right:
            seg = clv_segments.copy()
            # Support older saved summaries while preferring the precise, newly named column.
            if "positive_clv_customers" not in seg.columns and "paying" in seg.columns:
                seg = seg.rename(columns={"paying": "positive_clv_customers"})
            for col in ["total_clv_36m", "mean_clv_36m", "share_of_total_clv", "customers", "positive_clv_customers"]:
                if col in seg.columns:
                    seg[col] = pd.to_numeric(seg[col], errors="coerce")
            if {"segment", "total_clv_36m"}.issubset(seg.columns):
                seg = seg.sort_values("total_clv_36m", ascending=True)
                fig = px.bar(
                    seg, x="total_clv_36m", y="segment", orientation="h",
                    title="Total Estimated 36-Month CLV by Plan Segment",
                    labels={"total_clv_36m": "Estimated discounted revenue (₹)", "segment": "Plan / billing period"},
                )
                st.plotly_chart(fig, width="stretch")

        st.subheader("CLV by Subscription Segment")
        st.caption(
            "The average CLV values include all active subscriptions in each segment, "
            "including zero-valued free-plan and unbilled records. `positive_clv_customers` "
            "counts rows with positive estimated CLV; it does not prove that every such customer "
            "has a verified successful-payment history."
        )
        segment_display = clv_segments.copy()
        if "positive_clv_customers" not in segment_display.columns and "paying" in segment_display.columns:
            segment_display = segment_display.rename(columns={"paying": "positive_clv_customers"})
        st.dataframe(segment_display, width="stretch", hide_index=True)

        if not clv_survival.empty:
            st.subheader("Estimated Retention by Plan Segment")
            st.caption("Kaplan–Meier retention estimates from observed history. Values beyond observed follow-up rely on an assumed constant-hazard tail.")
            surv = clv_survival.copy()
            value_cols = [c for c in ["S(3m)", "S(6m)", "S(12m)", "S(24m)"] if c in surv.columns]
            if "segment" in surv.columns and value_cols:
                long = surv.melt(id_vars="segment", value_vars=value_cols, var_name="Horizon", value_name="Survival probability")
                long["Horizon"] = pd.Categorical(long["Horizon"], categories=value_cols, ordered=True)
                fig = px.line(long, x="Horizon", y="Survival probability", color="segment", markers=True,
                              title="Historical Survival Probability by Plan Segment")
                fig.update_yaxes(tickformat=".0%", range=[0, 1])
                st.plotly_chart(fig, width="stretch")
                st.dataframe(clv_survival, width="stretch", hide_index=True)

        if not clv_sensitivity.empty:
            st.subheader("Sensitivity to Forecast Horizon and Discount Rate")
            st.caption("The 36-month estimates change with the discount-rate assumption. Longer horizons add more extrapolation risk.")
            st.dataframe(clv_sensitivity, width="stretch", hide_index=True)

        if "data_quality_flag" in cc.columns:
            flag_count = cc["data_quality_flag"].fillna("").astype(str).str.strip().ne("").sum()
            if flag_count:
                st.info(
                    f"Data-quality flag: {flag_count:,} active subscriptions have a paid plan but no historical billing records. "
                    "Their CLV is set to zero under the primary conservative rule. Missing records do not prove that future "
                    "value is zero, so the table below shows clearly labelled sensitivity scenarios."
                )

        if not clv_unbilled_breakdown.empty:
            st.subheader("Why Are Paid-Plan Subscriptions Unbilled?")
            st.caption("Most flagged records are a data gap (annual plans started in 2025, whose payments are missing), not Free→Basic upgrades.")
            st.dataframe(clv_unbilled_breakdown, width="stretch", hide_index=True)

        if not clv_unbilled_sensitivity.empty:
            st.subheader("Sensitivity to Missing Billing History")
            st.caption(
                "The alternative scenarios assume flagged subscriptions may have been billable. They use observed billing-period "
                "payment-success rates and the fitted survival profile for comparable plan/billing-period customers; the stress "
                "case assumes 100% payment success. These are assumption-based scenarios, not confidence intervals or verified revenue."
            )
            st.dataframe(clv_unbilled_sensitivity, width="stretch", hide_index=True)

        st.subheader("Example Customer-Level Estimates")
        display_cols = [c for c in ["subscription_id", "plan_name", "billing_period", "plan_price", "clv_12m", "clv_36m", "retention_prob_12m", "expected_active_months_36m", "data_quality_flag"] if c in cc.columns]
        if display_cols:
            top = cc.sort_values("clv_36m", ascending=False)[display_cols].head(20)
            st.dataframe(top, width="stretch", hide_index=True)

        if clv_audit:
            st.caption(
                f"Data audit: {clv_audit.get('rows', {}).get('subscriptions', '—'):,} subscriptions in the source data; "
                f"last recorded payment date {clv_audit.get('last_payment_date', 'not available')}."
            )

        if clv_revenue_backtest.empty or clv_retention_backtest.empty:
            st.info(
                "Corrected historical survival backtests are pending. The backtest tables "
                "from the uploaded package were not carried over because the prior code "
                "could assign final plan IDs to historical snapshots. After installing "
                "requirements, run `python run_pipeline.py` to regenerate and enable those results."
            )
        else:
            st.subheader("Corrected Out-of-Time Backtests")
            st.caption("These results are generated by the corrected pipeline and remain separate from the 90-day XGBoost rolling validation above.")
            st.dataframe(clv_revenue_backtest, width="stretch", hide_index=True)
            st.dataframe(clv_retention_backtest, width="stretch", hide_index=True)

        if not clv_customer_grouped_holdout.empty:
            st.subheader("Customer-Grouped Holdout: Unseen Customers")
            st.caption(
                "Fixed cutoff: 30 June 2025; 3-month target horizon. Training rows come only from earlier snapshots whose "
                "target windows had ended by the cutoff. Holdout customer IDs are removed from every historical training snapshot. "
                "Because this dataset has one subscription per customer, the key added safeguard is excluding each holdout customer "
                "from all historical training rows. This complements rolling temporal backtesting and does not replace real-world validation."
            )
            st.dataframe(clv_customer_grouped_holdout, width="stretch", hide_index=True)
        else:
            st.info(
                "Customer-grouped holdout results are not available yet. Run `python run_pipeline.py` to generate "
                "the fixed-cutoff evaluation on customers excluded from historical training snapshots."
            )

        st.warning(
            "Limitations: the source dataset appears synthetic; annual-plan renewals "
            "are missing from 2025; customer attributes add little detectable churn "
            "signal; and estimates beyond observed follow-up rely on a constant-hazard "
            "extrapolation. Treat this as an academic methodology prototype, not a "
            "validated business valuation model."
        )


# ============================================================
# PAGE 7: RETENTION RECOMMENDATIONS
# ============================================================

elif page == "Retention Recommendations":
    st.subheader("Customer Value, Churn Risk and Retention Priorities")
    st.write(
        "This page combines the survival model's estimated 36-month revenue-based CLV with "
        "12-month churn risk (1 minus the Cox-model retention probability). Segments and actions "
        "are rule-based prioritization suggestions, not tested evidence that a campaign will reduce churn."
    )
    st.caption(
        "High-value and high-risk mean at or above the median among paying customers with billing history. "
        "Free-plan customers form their own group, and paid-plan customers without billing history are separated "
        "for data review rather than being treated as confirmed zero-value customers. Plan drives both value and risk, "
        "so these segments largely reflect plan mix."
    )
    if clv_value_risk_summary.empty:
        st.info("Retention-priority outputs are not available. Run `python run_pipeline.py` to generate them.")
    else:
        summary = clv_value_risk_summary.copy()
        for col in ["customers", "total_estimated_clv_36m", "mean_estimated_clv_36m", "mean_estimated_12m_churn_risk", "flagged_billing_records"]:
            if col in summary.columns:
                summary[col] = pd.to_numeric(summary[col], errors="coerce")
        total_customers = int(summary["customers"].sum()) if "customers" in summary.columns else 0
        total_value = float(summary["total_estimated_clv_36m"].sum()) if "total_estimated_clv_36m" in summary.columns else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("Scored active subscriptions", f"{total_customers:,}")
        c2.metric("Conservative estimated 36-month CLV", money(total_value))
        risk_count = int(summary.loc[summary["value_risk_segment"] == "High value / high risk", "customers"].sum()) if {"value_risk_segment", "customers"}.issubset(summary.columns) else 0
        c3.metric("High-value / high-risk customers", f"{risk_count:,}")

        if {"value_risk_segment", "customers"}.issubset(summary.columns):
            st.plotly_chart(px.bar(summary.sort_values("customers"), x="customers", y="value_risk_segment", orientation="h", title="Customer Counts by Value–Risk Segment", labels={"value_risk_segment": "Segment", "customers": "Customers"}), width="stretch")
        if {"value_risk_segment", "total_estimated_clv_36m"}.issubset(summary.columns):
            st.plotly_chart(px.bar(summary.sort_values("total_estimated_clv_36m"), x="total_estimated_clv_36m", y="value_risk_segment", orientation="h", title="Conservative Estimated CLV by Value–Risk Segment", labels={"value_risk_segment": "Segment", "total_estimated_clv_36m": "Estimated discounted revenue (₹)"}), width="stretch")
        st.dataframe(summary, width="stretch", hide_index=True)

        if not clv_retention_recommendations.empty:
            st.subheader("Customer-Level Suggested Actions")
            st.caption("Only the higher-priority / billing-review cohort is included in this table. Review flagged billing histories before using them to make decisions.")
            recs = clv_retention_recommendations.copy()
            display_cols = [c for c in ["subscription_id", "plan_name", "billing_period", "clv_36m", "estimated_12m_churn_risk", "value_risk_segment", "data_quality_flag", "retention_recommendation"] if c in recs.columns]
            if "clv_36m" in recs.columns:
                recs["clv_36m"] = pd.to_numeric(recs["clv_36m"], errors="coerce")
            if "estimated_12m_churn_risk" in recs.columns:
                recs["estimated_12m_churn_risk"] = pd.to_numeric(recs["estimated_12m_churn_risk"], errors="coerce")
            st.dataframe(recs[display_cols].head(100), width="stretch", hide_index=True)
        else:
            st.info("No priority-list rows are available in the current generated outputs.")
    st.warning("The dataset is synthetic. These risk scores and recommendations are research outputs, not validated production decisions; retention campaigns should be evaluated before claiming impact.")

elif page == "Uncertainty & Scenarios":
    st.subheader("Uncertainty Around CLV Projections and Scenario Simulation")
    st.write(
        "Point estimates hide how uncertain a projection is. This page shows (1) bootstrap confidence intervals for "
        "CLV, (2) a Monte Carlo predictive interval for the revenue that will actually be collected, (3) an out-of-time "
        "check of whether those intervals contain what really happened, and (4) a what-if simulator."
    )
    st.caption(
        "Estimation uncertainty = how much the estimate would change with a different sample of the same size (bootstrap). "
        "Process uncertainty = customers churn and payments fail at random even if the model were perfect (Monte Carlo). "
        "Neither covers model misspecification, future shocks or the missing-billing assumption."
    )
    if clv_unc_total.empty:
        st.info("Uncertainty outputs are not available. Run `python run_pipeline.py` to generate them.")
    else:
        est = clv_unc_total.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Point estimate (36-month CLV)", money(est["point_estimate"]))
        c2.metric("95% confidence interval", f"{money(est['ci_low'])} – {money(est['ci_high'])}")
        if len(clv_unc_total) > 1:
            pred = clv_unc_total.iloc[1]
            c3.metric("95% predictive interval", f"{money(pred['ci_low'])} – {money(pred['ci_high'])}")
        st.dataframe(clv_unc_total, width="stretch", hide_index=True)

        if not clv_unc_reps.empty and not clv_unc_draws.empty:
            fig = go.Figure()
            fig.add_histogram(x=clv_unc_draws["simulated_revenue_36m"], name="Predictive (estimation + random churn/payments)", opacity=0.5, nbinsx=40, histnorm="probability density")
            fig.add_histogram(x=clv_unc_reps["total_clv_36m"], name="Bootstrap (estimation only)", opacity=0.75, nbinsx=25, histnorm="probability density")
            fig.add_vline(x=float(est["point_estimate"]), line_dash="dash", annotation_text="Point estimate")
            fig.update_layout(barmode="overlay", title="Distribution of Portfolio 36-Month Revenue",
                              xaxis_title="Discounted revenue (₹)", yaxis_title="Density")
            st.plotly_chart(fig, width="stretch")

        if not clv_unc_segments.empty:
            st.subheader("Segment-Level 95% Bootstrap Intervals")
            seg = clv_unc_segments.copy()
            fig = go.Figure(go.Bar(
                x=seg["total_clv_36m"], y=seg["segment"], orientation="h",
                error_x=dict(type="data", symmetric=False,
                             array=(seg["clv_ci_high"] - seg["total_clv_36m"]).clip(lower=0),
                             arrayminus=(seg["total_clv_36m"] - seg["clv_ci_low"]).clip(lower=0)),
            ))
            fig.update_layout(title="Total 36-Month CLV by Segment with 95% Interval",
                              xaxis_title="Discounted revenue (₹)", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(seg, width="stretch", hide_index=True)

        st.subheader("Do the Predictive Intervals Contain Reality?")
        if clv_interval_coverage.empty:
            st.info("Interval backtest not available (the pipeline was run with --skip-backtest).")
        else:
            cov = clv_interval_coverage.copy()
            inside, total_n = int(cov["actual_within_interval"].sum()), len(cov)
            st.metric("Realised outcomes inside the 90% interval", f"{inside} of {total_n}", help="Nominal coverage is 90%. Few origins, so this is a rough check.")
            st.caption("Each row is a past forecast origin: an interval built only from data available then, compared with revenue actually collected from paid monthly customers.")
            st.dataframe(cov, width="stretch", hide_index=True)

    st.subheader("Scenario Simulator")
    if clv_scenarios.empty:
        st.info("Scenario grid not available. Run `python run_pipeline.py` to generate it.")
    else:
        g = clv_scenarios
        base_key = dict(churn_hazard_change_pct=0, annual_discount_rate=0.10, horizon_months=36, payment_success_multiplier=1.0)

        def _grid_total(**kw):
            m = np.ones(len(g), dtype=bool)
            for k, v in kw.items():
                m &= np.isclose(g[k].astype(float), float(v))
            return g[m]

        st.caption("Pick what-if levers. Results are point estimates for the 31 Dec 2025 active customers with billing evidence. "
                   "A churn change of −25% multiplies every customer's churn hazard by 0.75.")
        s1, s2 = st.columns(2)
        with s1:
            dh = st.select_slider("Change in churn hazard (%)", options=sorted(g["churn_hazard_change_pct"].unique().tolist()), value=0)
            hz = st.select_slider("Forecast horizon (months)", options=sorted(g["horizon_months"].unique().tolist()), value=36)
        with s2:
            dr = st.select_slider("Annual discount rate", options=sorted(g["annual_discount_rate"].unique().tolist()), value=0.10,
                                  format_func=lambda x: f"{x:.0%}")
            sm = st.select_slider("Payment success vs observed", options=sorted(g["payment_success_multiplier"].unique().tolist()), value=1.0,
                                  format_func=lambda x: f"{x:.0%}")
        chosen = _grid_total(churn_hazard_change_pct=dh, annual_discount_rate=dr, horizon_months=hz, payment_success_multiplier=sm)
        base = _grid_total(**base_key)
        horizon_matched_base = _grid_total(churn_hazard_change_pct=0, annual_discount_rate=dr, horizon_months=hz, payment_success_multiplier=1.0)
        if chosen.empty or base.empty:
            st.warning("The selected combination is not in the scenario grid.")
        else:
            scen_total, base_total = float(chosen["total_clv"].sum()), float(base["total_clv"].sum())
            ref_total = float(horizon_matched_base["total_clv"].sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Scenario portfolio CLV", money(scen_total), delta=f"{(scen_total / base_total - 1) * 100:+.1f}% vs default (36m, 10%, observed churn)")
            m2.metric("Default scenario", money(base_total))
            m3.metric("Effect of the levers alone", money(scen_total - ref_total), help="Scenario vs the same horizon and discount rate with observed churn and payment success.")
            comp = pd.DataFrame({"segment": chosen["segment"].values, "Scenario": chosen["total_clv"].values}).merge(
                horizon_matched_base[["segment", "total_clv"]].rename(columns={"total_clv": "Observed churn / payments"}), on="segment")
            comp = comp.melt(id_vars="segment", var_name="case", value_name="total_clv")
            st.plotly_chart(px.bar(comp, x="total_clv", y="segment", color="case", barmode="group", orientation="h",
                                   title="CLV by Segment: Scenario vs Observed (same horizon and discount rate)",
                                   labels={"total_clv": "Discounted revenue (₹)", "segment": "Segment"}), width="stretch")

            st.subheader("Which Lever Matters Most? (one at a time, vs default)")
            rows = []
            for d in sorted(g["churn_hazard_change_pct"].unique()):
                rows.append({"lever": "Churn hazard change (%)", "setting": f"{d:+d}%",
                             "portfolio_clv": float(_grid_total(**{**base_key, "churn_hazard_change_pct": d})["total_clv"].sum())})
            for d in sorted(g["annual_discount_rate"].unique()):
                rows.append({"lever": "Annual discount rate", "setting": f"{d:.0%}",
                             "portfolio_clv": float(_grid_total(**{**base_key, "annual_discount_rate": d})["total_clv"].sum())})
            for d in sorted(g["horizon_months"].unique()):
                rows.append({"lever": "Horizon (months)", "setting": f"{int(d)}",
                             "portfolio_clv": float(_grid_total(**{**base_key, "horizon_months": d})["total_clv"].sum())})
            for d in sorted(g["payment_success_multiplier"].unique()):
                rows.append({"lever": "Payment success vs observed", "setting": f"{d:.0%}",
                             "portfolio_clv": float(_grid_total(**{**base_key, "payment_success_multiplier": d})["total_clv"].sum())})
            lever = pd.DataFrame(rows)
            lever["change_vs_default_pct"] = 100 * (lever["portfolio_clv"] / base_total - 1)
            st.dataframe(lever, width="stretch", hide_index=True)
    st.warning("Scenarios are mechanical what-ifs on a fitted model, not forecasts of what a retention programme will achieve. "
               "The dataset is synthetic; real intervals would be wider.")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SubscriptIQ · Academic research prototype · Historical data only · "
    "Not financial advice or a production customer decision system"
)
