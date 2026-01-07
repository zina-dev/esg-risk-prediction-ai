# =====================================================================
# feature_engineering_updated.py   (FINAL CLEAN VERSION — 2025)
# =====================================================================

import pandas as pd
import numpy as np
from dowhy import CausalModel



# =====================================================================
# 1. MISSING VALUE IMPUTATION
# =====================================================================

def impute_missing_values(df, columns_to_impute):
    """Impute numeric columns with median."""
    if df is None:
        print("⚠️ No data to impute.")
        return None

    df_out = df.copy()
    print("\n📌 Imputing missing values...")

    for col in columns_to_impute:
        if col in df_out.columns and pd.api.types.is_numeric_dtype(df_out[col]):
            med = df_out[col].median()
            df_out[col] = df_out[col].fillna(med)
            print(f"  → Filled {col} with median = {med:.4f}")
        else:
            print(f"  → Skipped (not numeric or missing): {col}")

    return df_out



# =====================================================================
# 2. PREPROCESS DATA (LOG + OHE) — MR features added LATER
# =====================================================================

def preprocess_data(df, log_transform_cols, ohe_cols):
    """
    - Clean Region names
    - Log-transform numeric columns
    - One-hot encode categorical variables
    """
    print("\n🚀 Starting preprocessing...")
    processed = df.copy()

    # Clean region string
    if "Region" in processed.columns:
        processed["Region"] = (
            processed["Region"]
            .astype(str)
            .str.strip()
            .str.replace("_", " ")
            .str.replace("-", " ")
            .str.title()
        )
        processed["Region"] = processed["Region"].replace(
            ["", "Na", "Nan", "None"], "Unknown"
        )

    # Log transform
    print(f"📌 Applying log1p to: {log_transform_cols}")
    for col in log_transform_cols:
        if col in processed.columns:
            processed[col] = np.log1p(processed[col])

    # Ensure Region is included in OHE
    if "Region" not in ohe_cols:
        ohe_cols.append("Region")

    print(f"📌 One-hot encoding: {ohe_cols}")
    processed = pd.get_dummies(processed, columns=ohe_cols, drop_first=True)

    print("✅ Preprocessing completed.")
    return processed



# =====================================================================
# 3. DYNAMIC RULE-BASED COMPLIANCE + PROFIT SIZE
# =====================================================================

REGION_TIERS = {
    "Europe": 3,
    "North America": 2,
    "Oceania": 2,
    "Asia": 1,
    "Latin America": 1,
    "Middle East": 2,
}

def _dynamic_rule_scores(df):
    """Returns region-tier scaled violation score."""
    df = df.copy()

    pm_q75 = df.groupby(["Industry", "Region"])["ProfitMargin"].transform(lambda s: s.quantile(0.75))
    esg_q25 = df.groupby(["Region"])["ESG_Overall"].transform(lambda s: s.quantile(0.25))

    high_pm = df["ProfitMargin"] >= pm_q75
    low_esg = df["ESG_Overall"] <= esg_q25

    base = (high_pm & low_esg).astype(int)
    region_tier = df["Region"].map(lambda r: REGION_TIERS.get(r, 1))

    return base * region_tier


def add_dynamic_compliance_and_profit_size(df_original, ohe_df):
    """
    Adds:
      - compliance_violation_score (quantile rule)
      - profit_per_size
      - reconstructs Industry + Region from OHE for causal graph
    """
    fused = ohe_df.copy()

    # Rebuild Industry
    industry_cols = [c for c in fused.columns if c.startswith("Industry_")]
    region_cols = [c for c in fused.columns if c.startswith("Region_")]

    fused["Industry"] = fused[industry_cols].idxmax(axis=1).str.replace("Industry_", "")
    fused["Region"]   = fused[region_cols].idxmax(axis=1).str.replace("Region_", "")

    # Inject numeric originals
    fused[["ProfitMargin", "ESG_Overall", "Revenue", "MarketCap"]] = \
        df_original[["ProfitMargin", "ESG_Overall", "Revenue", "MarketCap"]].values

    # Compliance Score
    fused["compliance_violation_score"] = _dynamic_rule_scores(fused)

    # Profit per size
    fused["profit_per_size"] = np.where(
        fused["MarketCap"] > 0,
        fused["Revenue"] / fused["MarketCap"],
        0.0
    )

    print("✅ Dynamic compliance + profit_per_size added.")
    return fused



# =====================================================================
# 4. CAUSAL ATE FEATURES (ProfitMargin → ESG)
# =====================================================================

def compute_pm_to_esg_ate(df):
    """Compute ATE using DoWhy."""
    graph_str = r"""
    digraph {
        Industry -> Revenue;
        Industry -> ProfitMargin;
        Region -> Revenue;
        Region -> ESG_Overall;
        Revenue -> ProfitMargin;
        Revenue -> MarketCap;
        Revenue -> ESG_Overall;
        MarketCap -> ESG_Overall;
        GrowthRate -> ESG_Overall;
        ProfitMargin -> ESG_Overall;
    }
    """

    df_causal = df.dropna(subset=["ProfitMargin", "ESG_Overall"])

    model = CausalModel(
        data=df_causal,
        treatment="ProfitMargin",
        outcome="ESG_Overall",
        graph=graph_str
    )

    ident = model.identify_effect(proceed_when_unidentifiable=True)

    est = model.estimate_effect(
        ident,
        method_name="backdoor.linear_regression",
        test_significance=True
    )

    return float(est.value)


def add_causal_features(df_original, df_dynamic):
    """Add ATE and pm_times_ate."""
    fused = df_dynamic.copy()
    ate = compute_pm_to_esg_ate(df_original)

    fused["causal_ate_pm_to_esg"] = ate
    fused["pm_times_ate"] = fused["ProfitMargin"] * ate

    print(f"✅ Added causal ATE features (ATE = {ate:.4f})")
    return fused



# =====================================================================
# 5. FINAL FUSION BUILDER
# =====================================================================

def add_fusion_features_final(
    df_original,
    df_dynamic,          # IMPORTANT: from dynamic + causal
    graph_embeddings,
    compliance_score=None
):
    """
    Final Fusion Dataset = 
      - df_dynamic (OHE + dynamic rule + causal)
      - KG embeddings (CompanyID)
      - median rule compliance score
    """
    fused = df_dynamic.copy()

    # Ensure CompanyID for merge
    fused["CompanyID"] = df_original["CompanyID"].astype(str)

    # --- KG Embeddings ---
    if graph_embeddings is not None:
        ge = graph_embeddings.copy()
        ge["CompanyID"] = ge["CompanyID"].astype(str)
        fused = fused.merge(ge, on="CompanyID", how="left")

    # --- Add median rule score if provided ---
    if (compliance_score is not None) and ("compliance_violation_score" not in fused.columns):
        fused["compliance_violation_score"] = compliance_score

    fused = fused.fillna(0)

    print(f"🎉 Final fused dataset created with {fused.shape[1]} columns.")
    return fused
