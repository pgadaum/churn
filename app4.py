"""
Telco Customer Churn Risk Predictor
DATA 4382 Capstone II — Pawan Gadaum & Saminas Kebebe

Self-contained: downloads data and trains the model on first load.
No churn_model.pkl required — works on Streamlit Cloud.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Churn Risk Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Force Light Mode Background & Default Text Color */
    [data-testid="stAppViewContainer"] {
        background-color: #F4F6F9;
        color: #1a1a1a;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    .header-bar {
        background: linear-gradient(90deg, #1B3564 0%, #0E7B8C 100%);
        padding: 1.2rem 2rem; border-radius: 8px; margin-bottom: 1.2rem;
    }
    .header-bar h1 { color:#fff; font-size:1.6rem; margin:0; }
    .header-bar p  { color:#A0C4E0; font-size:0.82rem; margin:0.2rem 0 0 0; }
    .risk-high   { background:#C0392B; color:#fff; padding:0.45rem 1.1rem;
                   border-radius:6px; font-size:1.05rem; font-weight:700; display:inline-block; }
    .risk-medium { background:#D97706; color:#fff; padding:0.45rem 1.1rem;
                   border-radius:6px; font-size:1.05rem; font-weight:700; display:inline-block; }
    .risk-low    { background:#1A7A4A; color:#fff; padding:0.45rem 1.1rem;
                   border-radius:6px; font-size:1.05rem; font-weight:700; display:inline-block; }
    .lbl { font-size:0.72rem; font-weight:700; letter-spacing:2px;
           color:#0E7B8C; text-transform:uppercase; margin:1.2rem 0 0.3rem 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training model on first load — takes ~30 seconds…")
def build_model():
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from imblearn.combine import SMOTEENN
    from imblearn.pipeline import Pipeline as ImbPipeline
    import kagglehub

    # 1. Load and clean data
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    df = pd.read_csv(path + "/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    feature_cols = [c for c in df.columns if c not in ["customerID", "Churn"]]
    cat_cols     = [c for c in feature_cols if df[c].dtype == "object"]
    num_cols     = [c for c in feature_cols if df[c].dtype != "object"]

    X = df[feature_cols]
    y = df["Churn"]

    # 2. Encode categorical variables BEFORE the pipeline to satisfy CalibratedClassifierCV
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    
    X_encoded = preprocessor.fit_transform(X)
    all_feat_names = preprocessor.get_feature_names_out()
    X_encoded_df = pd.DataFrame(X_encoded, columns=all_feat_names)

    # 3. Split the already-encoded data
    X_train, _, y_train, _ = train_test_split(X_encoded_df, y, test_size=0.20, random_state=42, stratify=y)

    # 4. Pipeline now only handles scaling and SMOTE to keep it perfectly leak-proof
    pipe = ImbPipeline([
        ("scale",    StandardScaler()),
        ("smoteenn", SMOTEENN(random_state=42)),
        ("clf",      GradientBoostingClassifier(
                         n_estimators=180, max_depth=3, learning_rate=0.08,
                         subsample=0.85, min_samples_leaf=12, random_state=42)),
    ])

    # 5. Train and calibrate
    calibrated = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)

    # 6. Extract components for Streamlit and SHAP
    base_pipe     = calibrated.calibrated_classifiers_[0].estimator
    fitted_scaler = base_pipe.named_steps["scale"]
    fitted_clf    = base_pipe.named_steps["clf"]
    explainer     = shap.TreeExplainer(fitted_clf)

    return dict(model=calibrated, feature_cols=feature_cols,
                all_feat_names=all_feat_names, preprocessor=preprocessor,
                fitted_scaler=fitted_scaler, explainer=explainer, threshold=0.38)


try:
    b = build_model()
except Exception as e:
    st.error(f"**Model training failed:** {e}")
    st.stop()

model          = b["model"]
feature_cols   = b["feature_cols"]
all_feat_names = b["all_feat_names"]
preprocessor   = b["preprocessor"]
fitted_scaler  = b["fitted_scaler"]
explainer      = b["explainer"]
THRESHOLD      = b["threshold"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <h1>🔮 Churn Risk Predictor</h1>
  <p>DATA 4382 Capstone II &nbsp;|&nbsp; Pawan Gadaum &amp; Saminas Kebebe &nbsp;|&nbsp; University of Texas Arlington</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Customer Profile")
    st.markdown("---")
    st.markdown("**Account**")
    contract     = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    tenure       = st.slider("Tenure (months)", 0, 72, 6)
    payment_meth = st.selectbox("Payment Method",
                    ["Electronic check", "Mailed check",
                     "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless    = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)
    st.markdown("---")
    st.markdown("**Services**")
    internet     = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_sec   = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
    online_bkp   = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
    device_prot  = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
    streaming_mv = st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])
    phone_svc    = st.radio("Phone Service", ["Yes", "No"], horizontal=True)
    multi_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    st.markdown("---")
    st.markdown("**Demographics & Billing**")
    gender       = st.radio("Gender", ["Male", "Female"], horizontal=True)
    senior       = st.radio("Senior Citizen", ["No", "Yes"], horizontal=True)
    partner      = st.radio("Partner", ["Yes", "No"], horizontal=True)
    dependents   = st.radio("Dependents", ["No", "Yes"], horizontal=True)
    monthly_chg  = st.slider("Monthly Charges ($)", 18.0, 120.0, 72.5, step=0.5)
    total_chg    = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0,
                    value=float(round(monthly_chg * max(tenure, 1), 2)), step=1.0)
    st.markdown("---")
    predict_btn = st.button("▶  Predict Churn Risk", type="primary", use_container_width=True)

input_df = pd.DataFrame([{
    "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner, "Dependents": dependents, "tenure": tenure,
    "PhoneService": phone_svc, "MultipleLines": multi_lines,
    "InternetService": internet, "OnlineSecurity": online_sec,
    "OnlineBackup": online_bkp, "DeviceProtection": device_prot,
    "TechSupport": tech_support, "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_mv, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment_meth,
    "MonthlyCharges": monthly_chg, "TotalCharges": total_chg,
}])[feature_cols]

def get_segment(prob, tenure, monthly, contract):
    if prob > 0.50:
        return "🔴 High-Risk", "Immediate outreach — offer a contract upgrade incentive or targeted discount before this customer cancels."
    elif tenure > 45 and prob < 0.20:
        return "🟢 Loyal & Stable", "Reward & cross-sell — premium tier upgrades, referral programs, recognition for loyalty."
    elif monthly > 70:
        return "🟠 High-Value At-Risk", "Price & value framing — demonstrate ROI of staying; offer a bundle deal or loyalty credit."
    else:
        return "🟡 Mid-Tier", "Engagement & upsell — loyalty rewards, add-on security or support services."

# ── Landing page ──────────────────────────────────────────────────────────────
if not predict_btn:
    st.markdown('<p class="lbl">How to Use This Tool</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;gap:1rem;margin-bottom:1.2rem;">
      <div style="flex:1;background:#1B3564;border-radius:8px;padding:1rem 1.2rem;">
        <div style="color:#C9941A;font-weight:700;font-size:0.95rem;margin-bottom:0.4rem;">Step 1 — Fill the form</div>
        <div style="color:#D0E4F5;font-size:0.88rem;">Enter the customer's contract type, tenure, services, and billing details in the left sidebar.</div>
      </div>
      <div style="flex:1;background:#1B3564;border-radius:8px;padding:1rem 1.2rem;">
        <div style="color:#C9941A;font-weight:700;font-size:0.95rem;margin-bottom:0.4rem;">Step 2 — Click Predict</div>
        <div style="color:#D0E4F5;font-size:0.88rem;">The calibrated GBT model scores the customer and returns a churn probability with SHAP explanations.</div>
      </div>
      <div style="flex:1;background:#1B3564;border-radius:8px;padding:1rem 1.2rem;">
        <div style="color:#C9941A;font-weight:700;font-size:0.95rem;margin-bottom:0.4rem;">Step 3 — Review &amp; Act</div>
        <div style="color:#D0E4F5;font-size:0.88rem;">Review the risk tier, top SHAP drivers, and recommended retention strategy for this customer.</div>
      </div>
    </div>""", unsafe_allow_html=True)

    def mcard(val, lbl, color):
        return (f'<div style="background:white;border-radius:8px;padding:1rem 1.2rem;'
                f'box-shadow:0 2px 8px rgba(0,0,0,0.08);text-align:center;">'
                f'<div style="font-size:2rem;font-weight:700;color:{color};">{val}</div>'
                f'<div style="font-size:0.8rem;color:#64748B;margin-top:0.2rem;">{lbl}</div></div>')
    st.markdown('<p class="lbl">Model at a Glance</p>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(mcard("0.849", "ROC-AUC (hold-out test)", "#1B3564"), unsafe_allow_html=True)
    m2.markdown(mcard("74.1%", "Recall — churners caught", "#0E7B8C"), unsafe_allow_html=True)
    m3.markdown(mcard("7,032", "Training customers", "#C9941A"), unsafe_allow_html=True)
    m4.markdown(mcard("0.38",  "Decision threshold", "#1A7A4A"), unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""<div style="background:#FEF3C7;border-left:4px solid #D97706;padding:0.7rem 1rem;
                border-radius:4px;font-size:0.85rem;color:#1A1A1A;">
    ⚠️ <strong>Limitations:</strong> This tool is a decision-support aid, not a replacement for human judgment.
    Scores reflect historical behavioral patterns and may not account for recent price changes, service
    disruptions, or competitive events.</div>""", unsafe_allow_html=True)
    st.stop()

# ── Predict ───────────────────────────────────────────────────────────────────
with st.spinner("Scoring…"):
    # Apply the OneHotEncoder to the Streamlit input
    input_encoded = preprocessor.transform(input_df)
    input_encoded_df = pd.DataFrame(input_encoded, columns=all_feat_names)

    # Predict using the calibrated model
    prob      = float(model.predict_proba(input_encoded_df)[0, 1])
    risk_flag = prob >= THRESHOLD
    
    # SHAP requires the explicitly scaled features to match how TreeExplainer was trained
    X_proc    = fitted_scaler.transform(input_encoded_df)
    X_proc_df = pd.DataFrame(X_proc, columns=all_feat_names)
    
    sv        = explainer.shap_values(X_proc_df)
    
    # Robust: handle list-of-arrays (old shap) or single array (new shap)
    shap_arr  = sv[1] if isinstance(sv, list) and len(sv) > 1 else (sv[0] if isinstance(sv, list) else sv)
    shap_vals = np.array(shap_arr).ravel()
    seg_label, seg_action = get_segment(prob, tenure, monthly_chg, contract)

if prob >= 0.60:
    tier, badge_cls, tier_color = "HIGH RISK", "risk-high",   "#C0392B"
elif prob >= 0.38:
    tier, badge_cls, tier_color = "MODERATE",  "risk-medium", "#D97706"
else:
    tier, badge_cls, tier_color = "LOW RISK",  "risk-low",    "#1A7A4A"

# ── Top row ───────────────────────────────────────────────────────────────────
left, mid, right = st.columns([1.3, 1.2, 1.5])

with left:
    st.markdown('<p class="lbl">Churn Probability</p>', unsafe_allow_html=True)
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(4.2, 2.6), facecolor="white")
    ax.set_facecolor("white")
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), lw=20, color="#E2E8F0", solid_capstyle="butt")
    theta_f = np.linspace(np.pi, np.pi - prob * np.pi, 300)
    ax.plot(np.cos(theta_f), np.sin(theta_f), lw=20, color=tier_color, solid_capstyle="butt")
    ax.text(0,  0.12, f"{prob:.1%}", ha="center", va="center",
            fontsize=26, fontweight="bold", color=tier_color)
    ax.text(0, -0.22, "Churn Probability", ha="center", va="center", fontsize=9, color="#64748B")
    ax.text(-1.05, -0.12, "0%",   ha="center", fontsize=8, color="#94A3B8")
    ax.text( 1.05, -0.12, "100%", ha="center", fontsize=8, color="#94A3B8")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.4, 1.15)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    st.pyplot(fig, use_container_width=True)
    plt.close()

with mid:
    st.markdown('<p class="lbl">Risk Assessment</p>', unsafe_allow_html=True)
    st.markdown(f'<span class="{badge_cls}">{tier}</span><br><br>', unsafe_allow_html=True)
    flag_color = "#1A7A4A" if risk_flag else "#64748B"
    flag_text  = "Yes ✓" if risk_flag else "No"
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.88rem;">
      <tr><td style="padding:5px 6px;color:#64748B;font-weight:600;">Probability</td>
          <td style="padding:5px 6px;color:#1a1a1a;font-weight:700;">{prob:.1%}</td></tr>
      <tr style="background:#F5F7FA;">
          <td style="padding:5px 6px;color:#64748B;font-weight:600;">Threshold</td>
          <td style="padding:5px 6px;color:#1a1a1a;">{THRESHOLD:.0%}</td></tr>
      <tr><td style="padding:5px 6px;color:#64748B;font-weight:600;">Flag for outreach</td>
          <td style="padding:5px 6px;color:{flag_color};font-weight:700;">{flag_text}</td></tr>
      <tr style="background:#F5F7FA;">
          <td style="padding:5px 6px;color:#64748B;font-weight:600;">Contract</td>
          <td style="padding:5px 6px;color:#1a1a1a;">{contract}</td></tr>
      <tr><td style="padding:5px 6px;color:#64748B;font-weight:600;">Tenure</td>
          <td style="padding:5px 6px;color:#1a1a1a;">{tenure} months</td></tr>
    </table>""", unsafe_allow_html=True)

with right:
    st.markdown('<p class="lbl">Customer Segment</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#1B3564;color:white;border-radius:8px;padding:1rem 1.1rem;margin-top:0.3rem;">
      <div style="color:#C9941A;font-weight:700;font-size:1rem;">{seg_label}</div>
      <div style="color:#D0E4F5;font-size:0.88rem;margin-top:0.4rem;">{seg_action}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── SHAP + interpretation ─────────────────────────────────────────────────────
shap_col, interp_col = st.columns([1.6, 1])

with shap_col:
    st.markdown('<p class="lbl">Top Churn Drivers — SHAP Explanation</p>', unsafe_allow_html=True)
    st.caption("Red = pushes toward churn · Teal = protects against churn · Length = strength")

    N = 12
    idx_sorted = np.argsort(np.abs(shap_vals))[::-1][:N][::-1]
    names_plot = [all_feat_names[i] for i in idx_sorted]
    vals_plot  = [float(shap_vals[i]) for i in idx_sorted]
    bar_colors = ["#C0392B" if v > 0 else "#0E7B8C" for v in vals_plot]

    def clean(n):
        n = n.replace("cat__", "").replace("num__", "")
        parts = n.split("_", 1)
        return f"{parts[0]}: {parts[1].replace('_',' ')}" if len(parts) > 1 else n.replace("_", " ")

    plt.style.use("default")
    fig2, ax2 = plt.subplots(figsize=(7, 4.4), facecolor="white")
    ax2.set_facecolor("white")
    ax2.barh(range(N), vals_plot, color=bar_colors, height=0.6, edgecolor="white")
    ax2.set_yticks(range(N))
    ax2.set_yticklabels([clean(n) for n in names_plot], fontsize=9.5, color="#374151")
    ax2.axvline(0, color="#CBD5E1", lw=0.8)
    ax2.set_xlabel("SHAP value (impact on churn probability)", fontsize=8.5, color="#64748B")
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.tick_params(left=False, axis="x", colors="#94A3B8", labelsize=8)
    fig2.tight_layout(pad=0.4)
    st.pyplot(fig2, use_container_width=True)
    plt.close()

with interp_col:
    st.markdown('<p class="lbl">Interpretation Guide</p>', unsafe_allow_html=True)

    push  = [(all_feat_names[i], float(shap_vals[i]))
             for i in np.argsort(shap_vals)[::-1][:3] if shap_vals[i] > 0]
    guard = [(all_feat_names[i], float(shap_vals[i]))
             for i in np.argsort(shap_vals)[:3] if shap_vals[i] < 0]

    if push:
        rows = "".join(
            f'<div style="padding:4px 0;color:#1a1a1a;font-size:0.87rem;">'
            f'🔴 <b>{feat.replace("_"," ")}</b>'
            f'<span style="color:#C0392B;"> (+{val:.3f})</span></div>'
            for feat, val in push)
        st.markdown(
            f'<div style="margin-bottom:0.6rem;">'
            f'<div style="font-weight:700;color:#1a1a1a;margin-bottom:3px;">Pushing toward churn:</div>'
            f'{rows}</div>', unsafe_allow_html=True)

    if guard:
        rows = "".join(
            f'<div style="padding:4px 0;color:#1a1a1a;font-size:0.87rem;">'
            f'🟢 <b>{feat.replace("_"," ")}</b>'
            f'<span style="color:#1A7A4A;"> ({val:.3f})</span></div>'
            for feat, val in guard)
        st.markdown(
            f'<div style="margin-bottom:0.8rem;">'
            f'<div style="font-weight:700;color:#1a1a1a;margin-bottom:3px;">Protecting against churn:</div>'
            f'{rows}</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:1px solid #e2e8f0;margin:0.5rem 0;">', unsafe_allow_html=True)
    st.markdown("""<div style="font-size:0.84rem;color:#374151;line-height:1.55;">
      <b>What SHAP values mean:</b><br>
      • Additive contribution of each feature to the final score<br>
      • Baseline = average prediction across all customers<br>
      • Sum of all values + baseline = final probability<br>
      • Larger bar = stronger influence on this prediction
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style="background:#FEF3C7;border-left:4px solid #D97706;padding:0.6rem 0.9rem;
                border-radius:4px;font-size:0.83rem;color:#1a1a1a;">
    ⚠️ <b>Note:</b> Always pair model output with agent judgment before contacting a customer.
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<small style='color:#94A3B8'>Calibrated GBT + SMOTE-ENN · Optuna-tuned · "
    "Threshold 0.38 · ROC-AUC 0.849 · DATA 4382 Capstone II · UTA</small>",
    unsafe_allow_html=True)
