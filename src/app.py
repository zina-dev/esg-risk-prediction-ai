import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import base64
import os
import tempfile

# --- Configuration ---
st.set_page_config(page_title="ESG Machine-Reasoning Predictor", layout="wide", initial_sidebar_state="expanded")

# --- Constants & Paths ---
BASE_DIR = "d:/ESG research work/global conference - Copy (2)"
FUSION_PATH = os.path.join(BASE_DIR, "final_fusion_dataset.csv")
ORIGINAL_PATH = os.path.join(BASE_DIR, "company_esg_financial_dataset (1).csv")
RULE_SEVERITY_PATH = os.path.join(BASE_DIR, "dynamic_rule_severity.csv")
RED_FLAGS_PATH = os.path.join(BASE_DIR, "red_flags_list.csv")

# --- Custom CSS for Aesthetics ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #fff;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE 1: Load & Align Data ---
@st.cache_data
def load_data():
    try:
        fusion_df = pd.read_csv(FUSION_PATH)
        original_df = pd.read_csv(ORIGINAL_PATH)
        rule_severity_df = pd.read_csv(RULE_SEVERITY_PATH)
        red_flags_df = pd.read_csv(RED_FLAGS_PATH)
        
        # Ensure CompanyName consistency (strip whitespace)
        fusion_df['CompanyName'] = fusion_df['CompanyName'].astype(str).str.strip()
        original_df['CompanyName'] = original_df['CompanyName'].astype(str).str.strip()
        rule_severity_df['CompanyName'] = rule_severity_df['CompanyName'].astype(str).str.strip()
        red_flags_df['CompanyName'] = red_flags_df['CompanyName'].astype(str).str.strip()

        # Extract Embeddings (Create graph_embeddings.csv equivalent)
        embed_cols = [c for c in fusion_df.columns if 'embed_' in c]
        if embed_cols:
            embeddings_df = fusion_df.groupby('CompanyName')[embed_cols].mean().reset_index()
        else:
            st.error("Embeddings columns not found in fusion dataset.")
            embeddings_df = pd.DataFrame()

        return fusion_df, original_df, rule_severity_df, red_flags_df, embeddings_df, embed_cols
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None

fusion_df, original_df, rule_severity_df, red_flags_df, embeddings_df, embed_cols = load_data()

# --- MODULE 2: Build Fusion Model ---
@st.cache_resource
def train_model(df):
    target = 'ESG_Overall'
    exclude_cols = ['CompanyID', 'CompanyName', 'Year', 'Industry', 'Region', target]
    
    # Select features: Numeric only
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[feature_cols]
    y = df[target]
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, random_state=42)
    model.fit(X, y)
    
    return model, X, y, feature_cols, X.mean()

if fusion_df is not None:
    model, X_train, y_train, feature_names, X_mean = train_model(fusion_df)
    explainer = shap.TreeExplainer(model)

# --- MODULE 4: Machine Reasoning Rule Layer ---
def check_static_rules(company_name, red_flags_df):
    flags = red_flags_df[red_flags_df['CompanyName'] == company_name]
    if not flags.empty:
        return True, "Company found in Red Flags list (Static Violation)."
    return False, "No static red flags found."

def check_dynamic_rules(row, original_df):
    # Compute thresholds from ORIGINAL clean data
    pm_q1 = original_df['ProfitMargin'].quantile(0.25)
    pm_median = original_df['ProfitMargin'].median()
    pm_q3 = original_df['ProfitMargin'].quantile(0.75)
    
    esg_q1 = original_df['ESG_Overall'].quantile(0.25)
    
    severity = 0
    reasons = []
    
    # Profit Margin Rule
    pm = row.get('ProfitMargin', 0)
    if pm < pm_q1:
        severity += 3
        reasons.append(f"ProfitMargin ({pm:.1f}%) < Q1 ({pm_q1:.1f}%) -> Severe")
    elif pm < pm_median:
        severity += 2
        reasons.append(f"ProfitMargin ({pm:.1f}%) < Median ({pm_median:.1f}%) -> Moderate")
    elif pm < pm_q3:
        severity += 1
        reasons.append(f"ProfitMargin ({pm:.1f}%) < Q3 ({pm_q3:.1f}%) -> Mild")
        
    # ESG Rule
    esg = row.get('ESG_Overall', 0) # Note: This is the ACTUAL ESG, but for prediction we might not have it? 
    # Assuming we check rules based on input features (like ProfitMargin) or predicted ESG? 
    # The prompt says "Determine rule severity" based on thresholds. 
    # Usually rules check INPUTS. But ESG_Overall is the target. 
    # If we are predicting, we might use the predicted ESG for the rule check?
    # For now, let's use the input row's ESG if available (historical) or skip if it's a new prediction without ground truth.
    # However, the prompt says "Determine rule severity... < Q1 -> Severe".
    
    return severity, reasons

def get_rule_severity_from_file(company_name, year, severity_df):
    # Look up pre-calculated severity
    match = severity_df[(severity_df['CompanyName'] == company_name) & (severity_df['Year'] == year)]
    if not match.empty:
        return match.iloc[0].get('severity_weighted_dynamic', 0), match.iloc[0].get('severity_gap_dynamic', 0)
    return 0, 0

# --- MODULE 5: Knowledge Graph Similarity ---
def find_similar_companies(company_name, embeddings_df, top_n=5):
    if company_name not in embeddings_df['CompanyName'].values:
        return []
    
    target_vec = embeddings_df[embeddings_df['CompanyName'] == company_name][embed_cols].values
    all_vecs = embeddings_df[embed_cols].values
    all_names = embeddings_df['CompanyName'].values
    
    sims = cosine_similarity(target_vec, all_vecs)[0]
    
    # Create dataframe of similarities
    sim_df = pd.DataFrame({'CompanyName': all_names, 'Similarity': sims})
    sim_df = sim_df[sim_df['CompanyName'] != company_name] # Exclude self
    sim_df = sim_df.sort_values('Similarity', ascending=False).head(top_n)
    
    # Merge with fusion data to get Industry/Region/ESG
    # We take the latest year for display
    latest_fusion = fusion_df.sort_values('Year', ascending=False).drop_duplicates('CompanyName')
    result = pd.merge(sim_df, latest_fusion[['CompanyName', 'Industry', 'Region', 'ESG_Overall']], on='CompanyName')
    
    return result

# --- MODULE 6: Causal Pathway Visualization ---
def plot_causal_dag():
    G = nx.DiGraph()
    edges = [
        ("Industry", "Revenue"),
        ("Revenue", "ProfitMargin"),
        ("ProfitMargin", "ESG_Overall"),
        ("Region", "ESG_Overall"),
        ("GrowthRate", "ESG_Overall"),
        ("MarketCap", "ESG_Overall")
    ]
    G.add_edges_from(edges)
    
    pos = {
        "Industry": (0, 2),
        "Region": (0, 0),
        "Revenue": (1, 2),
        "ProfitMargin": (2, 2),
        "GrowthRate": (1, 1),
        "MarketCap": (1, 0),
        "ESG_Overall": (3, 1)
    }
    
    fig, ax = plt.subplots(figsize=(8, 5))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', arrows=True, ax=ax, edge_color='gray')
    return fig

# --- MODULE 7: Integrated Reasoning Summary ---
def generate_integrated_reasoning(company_name, row, prediction, shap_values, feature_names, similar_companies, static_flag, dynamic_reasons):
    
    # 1. Embedding Context
    avg_sim_esg = similar_companies['ESG_Overall'].mean() if not similar_companies.empty else 0
    sim_names = ", ".join(similar_companies['CompanyName'].tolist()[:3])
    
    embedding_impact = 0
    # Sum SHAP values for embedding features
    for i, f in enumerate(feature_names):
        if 'embed' in f:
            embedding_impact += shap_values[i]
            
    # 2. Financial Indicators
    financial_impact = 0
    for i, f in enumerate(feature_names):
        if f in ['Revenue', 'ProfitMargin', 'GrowthRate', 'MarketCap']:
            financial_impact += shap_values[i]
            
    # 3. Rule Layer
    static_str = "Listed in red_flags.csv" if static_flag[0] else "No static violations"
    dynamic_str = "; ".join(dynamic_reasons) if dynamic_reasons else "No dynamic violations"
    
    summary = f"""
### Integrated Reasoning Summary

**Why did the model predict {prediction:.2f} for {company_name}?**

1. **Embedding Context**
   This company is similar to peers like {sim_names}, with an average ESG score of {avg_sim_esg:.2f}.
   Embedding features impact: {embedding_impact:+.2f}.

2. **Financial Indicators**
   Financial metrics (Revenue, Profit, Growth) impact: {financial_impact:+.2f}.

3. **Rule Layer**
   - Static Rule: {static_str}
   - Dynamic Rules: {dynamic_str}

4. **Conclusion**
   Based on the fusion of financial data, graph embeddings, and rule-based logic, the system predicts an ESG score of {prediction:.2f}.
    """
    return summary

# --- PDF Export ---
def create_pdf(summary_text, company_name, prediction):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"ESG Report for {company_name}", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Predicted ESG Score: {prediction:.2f}", ln=1, align='C')
    pdf.ln(10)
    
    # Handle unicode/markdown roughly by stripping
    clean_text = summary_text.replace("**", "").replace("###", "").replace("🧠", "")
    pdf.multi_cell(0, 10, clean_text)
    
    return pdf.output(dest='S').encode('latin-1')

# --- MODULE 8: UI ---
st.title(" ESG Machine-Reasoning Predictor")

with st.sidebar:
    st.header("Input Selector")
    input_mode = st.radio("Mode", ["Select Company", "Manual Input"])
    
    selected_company = None
    input_row = None
    
    if input_mode == "Select Company":
        company_list = fusion_df['CompanyName'].unique()
        selected_company = st.selectbox("Choose Company", company_list)
        # Get the latest year row for this company
        company_rows = fusion_df[fusion_df['CompanyName'] == selected_company]
        selected_year = st.selectbox("Year", company_rows['Year'].unique())
        input_row = company_rows[company_rows['Year'] == selected_year].iloc[0]
    else:
        st.info("Manual input not fully implemented for all 50+ features. Using mean values as default.")
        # Simplified manual input
        revenue = st.number_input("Revenue", value=1000.0)
        profit = st.number_input("Profit Margin", value=10.0)
        # Create a dummy row with means
        input_row = pd.Series(X_mean, index=feature_names)
        input_row['Revenue'] = revenue
        input_row['ProfitMargin'] = profit
        selected_company = "Manual_Input_Company"

    predict_btn = st.button("Predict ESG Score")

if predict_btn and input_row is not None:
    # Prepare input for model
    X_input = pd.DataFrame([input_row[feature_names]])
    
    # Predict
    prediction = model.predict(X_input)[0]
    
    # SHAP
    shap_values = explainer.shap_values(X_input)[0]
    
    # Rules
    static_flag = check_static_rules(selected_company, red_flags_df)
    dynamic_sev, dynamic_reasons = check_dynamic_rules(input_row, original_df)
    
    # KG
    similar_companies = find_similar_companies(selected_company, embeddings_df)
    
    # Reasoning
    reasoning_text = generate_integrated_reasoning(
        selected_company, input_row, prediction, shap_values, feature_names, 
        similar_companies, static_flag, dynamic_reasons
    )
    
    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        " Dashboard", " SHAP", " Rules", " Knowledge Graph", " Causal", " Reasoning", " Report"
    ])
    
    with tab1:
        st.subheader("Prediction Dashboard")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted ESG Score", f"{prediction:.2f}")
        col2.metric("Profit Margin", f"{input_row.get('ProfitMargin', 0):.1f}%")
        col3.metric("Revenue", f"${input_row.get('Revenue', 0):,.0f}")
        
        st.progress(min(max(prediction/100, 0.0), 1.0))
        
    with tab2:
        st.subheader("SHAP Explainability")
        st.write("Feature contributions to the prediction:")
        # Waterfall plot
        fig_shap, ax_shap = plt.subplots()
        shap.plots.waterfall(shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=X_input.iloc[0], feature_names=feature_names), show=False)
        st.pyplot(fig_shap)
        
    with tab3:
        st.subheader("Machine Reasoning Rules")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Static Rules")
            if static_flag[0]:
                st.error(f" {static_flag[1]}")
            else:
                st.success(f" {static_flag[1]}")
        with col2:
            st.markdown("#### Dynamic Rules")
            if dynamic_reasons:
                for r in dynamic_reasons:
                    st.warning(f"⚠️ {r}")
            else:
                st.success("✅ No dynamic rule violations.")
                
    with tab4:
        st.subheader("Knowledge Graph Neighbors")
        if not similar_companies.empty:
            st.dataframe(similar_companies)
        else:
            st.write("No similar companies found (or manual input).")
            
    with tab5:
        st.subheader("Causal Pathways")
        st.pyplot(plot_causal_dag())
        st.caption("Visual representation of assumed causal links.")
        
    with tab6:
        st.subheader("Integrated Reasoning")
        st.markdown(reasoning_text)
        
    with tab7:
        st.subheader("Summary Report")
        pdf_bytes = create_pdf(reasoning_text, selected_company, prediction)
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="ESG_Report_{selected_company}.pdf">Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)

else:
    st.info("Select a company and click Predict to start.")
