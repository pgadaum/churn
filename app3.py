"""
Telco Customer Churn Risk Predictor
DATA 4382 Capstone II — Pawan Gadaum & Saminas Kebebe

Streamlit app that loads a trained, calibrated GBT pipeline and:
  • Accepts a customer profile via sidebar form
  • Returns a calibrated churn probability + risk tier
  • Explains the top drivers with SHAP values
  • Shows the customer's segment and recommended retention action
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, pathlib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Risk Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .main { background-color: #F5F7FA; }
  /* Header strip */
  .header-bar {
    background: linear-gradient(90deg, #1B3564 0%, #0E7B8C 100%);
    padding: 1.2rem 2rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
  }
  .header-bar h1 { color: #FFFFFF; font-size: 1.7rem; margin: 0; font-family: 'Trebuchet MS'; }
  .header-bar p  { color: #A0C4E0; font-size: 0.85rem; margin: 0.2rem 0 0 0; }
  /* Risk badge */
  .badge-high   { background:#C0392B; color:white; padding:0.5rem 1.2rem; border-radius:6px;
                  font-size:1.1rem; font-weight:700; display:inline-block; }
  .badge-medium { background:#D97706; color:white; padding:0.5rem 1.2rem; border-radius:6px;
                  font-size:1.1rem; font-weight:700; display:inline-block; }
  .badge-low    { background:#1A7A4A; color:white; padding:0.5rem 1.2rem; border-radius:6px;
                  font-size:1.1rem; font-weight:700; display:inline-block; }
  /* Metric card */
  .metric-card {
    background: white; border-radius: 8px; padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
  }
  .metric-card .val { font-size: 2.2rem; font-weight: 700; font-family: 'Trebuchet MS'; }
  .metric-card .lbl { font-size: 0.8rem; color: #64748B; margin-top: 0.2rem; }
  /* Section divider */
  .section-hdr {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 2px;
    color: #0E7B8C; text-transform: uppercase; margin: 1.5rem 0 0.5rem 0;
  }
  /* Warning box */
  .warn-box {
    background: #FEF3C7; border-left: 4px solid #D97706;
    padding: 0.7rem 1rem; border-radius: 4px; font-size: 0.85rem; color: #1A1A1A;
  }
  /* Strategy card */
  .strategy-card {
    background: #1B3564; color: white; border-radius: 8px;
    padding: 1rem 1.2rem; margin-top: 0.5rem;
  }
  .strategy-card .seg-title { font-size: 1.0rem; font-weight: 700; color: #C9941A; }
  .strategy-card .seg-action { font-size: 0.9rem; color: #D0E4F5; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = pathlib.Path(__file__).parent / "churn_model.pkl"

@st.cache_resource(show_spinner="Loading model…")
def load_bundle(path):
    return joblib.load(path)

if not MODEL_PATH.exists():
    st.error(
        "**Model file not found.**  \n"
        "Run `python train_and_save.py` in this directory first to generate `churn_model.pkl`."
    )
    st.stop()

bundle = load_bundle(MODEL_PATH)
model          = bundle["model"]
feature_cols   = bundle["feature_cols"]
cat_cols       = bundle["cat_cols"]
num_cols       = bundle["num_cols"]
all_feat_names = bundle["all_feat_names"]
fitted_prep    = bundle["fitted_prep"]
fitted_clf     = bundle["fitted_clf"]
explainer      = bundle["explainer"]
THRESHOLD      = bundle["threshold"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <h1>🔮 Churn Risk Predictor</h1>
  <p>DATA 4382 Capstone II &nbsp;|&nbsp; Pawan Gadaum &amp; Saminas Kebebe &nbsp;|&nbsp; University of Texas Arlington</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: Customer Input Form ──────────────────────────────────────────────
st.sidebar.markdown("## 📋 Customer Profile")
st.sidebar.markdown("Enter the customer's details to score churn risk.")
st.sidebar.markdown("---")

with st.sidebar:
    st.markdown("**Account Info**")
    contract      = st.selectbox("Contract Type",
                      ["Month-to-month", "One year", "Two year"], index=0)
    tenure        = st.slider("Tenure (months)", 0, 72, 6)
    payment_meth  = st.selectbox("Payment Method",
                      ["Electronic check", "Mailed check",
                       "Bank transfer (automatic)", "Credit card (automatic)"],
                      index=0)
    paperless     = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)

    st.markdown("---")
    st.markdown("**Services Subscribed**")
    internet      = st.selectbox("Internet Service",
                      ["Fiber optic", "DSL", "No"], index=0)
    online_sec    = st.selectbox("Online Security",   ["No", "Yes", "No internet service"], index=0)
    online_bkp    = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"], index=0)
    device_prot   = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=0)
    tech_support  = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"], index=0)
    streaming_tv  = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"], index=0)
    streaming_mv  = st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"], index=0)
    phone_svc     = st.radio("Phone Service", ["Yes", "No"], horizontal=True)
    multi_lines   = st.selectbox("Multiple Lines",
                      ["No", "Yes", "No phone service"], index=0)

    st.markdown("---")
    st.markdown("**Demographics & Billing**")
    gender        = st.radio("Gender", ["Male", "Female"], horizontal=True)
    senior        = st.radio("Senior Citizen", ["No", "Yes"], horizontal=True)
    partner       = st.radio("Partner", ["Yes", "No"], horizontal=True)
    dependents    = st.radio("Dependents", ["No", "Yes"], horizontal=True)
    monthly_chg   = st.slider("Monthly Charges ($)", 18.0, 120.0, 72.5, step=0.5)
    total_chg     = st.number_input("Total Charges ($)",
                      min_value=0.0, max_value=9000.0,
                      value=float(round(monthly_chg * max(tenure, 1), 2)),
                      step=1.0)

    st.markdown("---")
    predict_btn = st.button("▶  Predict Churn Risk", type="primary", use_container_width=True)

# ── Build input DataFrame ─────────────────────────────────────────────────────
input_dict = {
    "gender":             gender,
    "SeniorCitizen":      1 if senior == "Yes" else 0,
    "Partner":            partner,
    "Dependents":         dependents,
    "tenure":             tenure,
    "PhoneService":       phone_svc,
    "MultipleLines":      multi_lines,
    "InternetService":    internet,
    "OnlineSecurity":     online_sec,
    "OnlineBackup":       online_bkp,
    "DeviceProtection":   device_prot,
    "TechSupport":        tech_support,
    "StreamingTV":        streaming_tv,
    "StreamingMovies":    streaming_mv,
    "Contract":           contract,
    "PaperlessBilling":   paperless,
    "PaymentMethod":      payment_meth,
    "MonthlyCharges":     monthly_chg,
    "TotalCharges":       total_chg,
}
input_df = pd.DataFrame([input_dict])[feature_cols]

# ── Segment logic ─────────────────────────────────────────────────────────────
def get_segment(prob, tenure, monthly, contract):
    if prob > 0.50:
        return ("🔴 High-Risk",
                "Immediate outreach — offer a contract upgrade incentive or targeted discount before this customer cancels.")
    elif tenure > 45 and prob < 0.20:
        return ("🟢 Loyal & Stable",
                "Reward & cross-sell — premium tier upgrades, referral programs, recognition for loyalty.")
    elif monthly > 70:
        return ("🟠 High-Value At-Risk",
                "Price & value framing — demonstrate ROI of staying; offer a bundle deal or loyalty credit.")
    else:
        return ("🟡 Mid-Tier",
                "Engagement & upsell — loyalty rewards, add-on security or support services.")

# ── Main panel: default state ─────────────────────────────────────────────────
if not predict_btn:
    st.markdown('<p class="lbl">How to Use This Tool</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; gap:1rem; margin-bottom:1rem;">
      <div style="flex:1; background:#1B3564; border-radius:8px; padding:1rem 1.2rem;">
        <div style="color:#C9941A; font-weight:700; font-size:0.95rem; margin-bottom:0.4rem;">Step 1 — Fill the form</div>
        <div style="color:#D0E4F5; font-size:0.88rem;">Enter the customer's contract type, tenure, services, and billing details in the left sidebar.</div>
      </div>
      <div style="flex:1; background:#1B3564; border-radius:8px; padding:1rem 1.2rem;">
        <div style="color:#C9941A; font-weight:700; font-size:0.95rem; margin-bottom:0.4rem;">Step 2 — Click Predict</div>
        <div style="color:#D0E4F5; font-size:0.88rem;">The calibrated GBT model scores the customer and returns a churn probability with confidence.</div>
      </div>
      <div style="flex:1; background:#1B3564; border-radius:8px; padding:1rem 1.2rem;">
        <div style="color:#C9941A; font-weight:700; font-size:0.95rem; margin-bottom:0.4rem;">Step 3 — Review &amp; Act</div>
        <div style="color:#D0E4F5; font-size:0.88rem;">Review the risk tier, top SHAP drivers, and the recommended retention strategy for this customer.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-hdr">Model at a Glance</p>', unsafe_allow_html=True)
    def mcard(val, lbl, color):
        return f"""<div style="background:white;border-radius:8px;padding:1rem 1.2rem;
                        box-shadow:0 2px 8px rgba(0,0,0,0.08);text-align:center;">
                  <div style="font-size:2rem;font-weight:700;color:{color};font-family:'Trebuchet MS';">{val}</div>
                  <div style="font-size:0.8rem;color:#64748B;margin-top:0.2rem;">{lbl}</div>
                </div>"""
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(mcard("0.849", "ROC-AUC (hold-out test)", "#1B3564"), unsafe_allow_html=True)
    with m2:
        st.markdown(mcard("74.1%", "Recall — churners caught", "#0E7B8C"), unsafe_allow_html=True)
    with m3:
        st.markdown(mcard("7,032", "Training customers", "#C9941A"), unsafe_allow_html=True)
    with m4:
        st.markdown(mcard("0.38", "Decision threshold", "#1A7A4A"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="warn-box">
    ⚠️ <strong>Limitations:</strong> This tool is a decision-support aid, not a replacement for human judgment. 
    Scores reflect historical behavioral patterns and may not account for recent price changes, 
    service disruptions, or competitive events. Retrain the model periodically to prevent drift.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── PREDICTION ────────────────────────────────────────────────────────────────
with st.spinner("Scoring customer…"):
    prob      = float(model.predict_proba(input_df)[0, 1])
    risk_flag = prob >= THRESHOLD

    # SHAP on base estimator
    X_proc    = fitted_prep.transform(input_df)
    X_proc_df = pd.DataFrame(X_proc, columns=all_feat_names)
    sv = explainer.shap_values(X_proc_df)
    # Handle all shap_values return shapes across sklearn/shap versions
    ev = explainer.expected_value
    ev_arr = np.array(ev).ravel()
    if len(ev_arr) > 1:
        base_val  = float(ev_arr[1])
        shap_vals = sv[1][0] if isinstance(sv, list) else sv[0]
    else:
        base_val  = float(ev_arr[0])
        shap_vals = sv[0] if not isinstance(sv, list) else sv[0][0]

    seg_label, seg_action = get_segment(prob, tenure, monthly_chg, contract)

# ── Risk tier ─────────────────────────────────────────────────────────────────
if prob >= 0.60:
    tier, badge_cls, tier_color = "HIGH RISK",   "badge-high",   "#C0392B"
elif prob >= 0.38:
    tier, badge_cls, tier_color = "MODERATE",    "badge-medium", "#D97706"
else:
    tier, badge_cls, tier_color = "LOW RISK",    "badge-low",    "#1A7A4A"

# ── Layout: top row ───────────────────────────────────────────────────────────
left, mid, right = st.columns([1.4, 1.2, 1.4])

with left:
    st.markdown('<p class="section-hdr">Churn Probability</p>', unsafe_allow_html=True)
    fig_gauge, ax = plt.subplots(figsize=(4.5, 2.8), facecolor="white")
    ax.set_facecolor("white")
    # Background arc
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), lw=22, color="#E2E8F0", solid_capstyle="butt")
    # Colored arc up to prob
    theta_fill = np.linspace(np.pi, np.pi - prob * np.pi, 200)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), lw=22, color=tier_color, solid_capstyle="butt")
    # Labels
    ax.text(0, 0.15, f"{prob:.1%}", ha="center", va="center", fontsize=28,
            fontweight="bold", color=tier_color, fontfamily="Trebuchet MS")
    ax.text(0, -0.22, "Churn Probability", ha="center", va="center",
            fontsize=9, color="#64748B")
    ax.text(-1.05, -0.15, "0%", ha="center", fontsize=8, color="#94A3B8")
    ax.text(1.05, -0.15, "100%", ha="center", fontsize=8, color="#94A3B8")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.4, 1.2)
    ax.axis("off")
    fig_gauge.tight_layout(pad=0.2)
    st.pyplot(fig_gauge, use_container_width=True)
    plt.close()

with mid:
    st.markdown('<p class="section-hdr">Risk Assessment</p>', unsafe_allow_html=True)
    st.markdown(f'<span class="{badge_cls}">{tier}</span>', unsafe_allow_html=True)
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
      <tr><td style="padding:6px 8px;color:#64748B;font-weight:600;">Probability</td>
          <td style="padding:6px 8px;color:#1a1a1a;font-weight:700;">{prob:.1%}</td></tr>
      <tr style="background:#F5F7FA;"><td style="padding:6px 8px;color:#64748B;font-weight:600;">Threshold</td>
          <td style="padding:6px 8px;color:#1a1a1a;">{THRESHOLD:.0%}</td></tr>
      <tr><td style="padding:6px 8px;color:#64748B;font-weight:600;">Flag for outreach</td>
          <td style="padding:6px 8px;color:{'#1A7A4A' if risk_flag else '#64748B'};font-weight:700;">{'Yes ✓' if risk_flag else 'No'}</td></tr>
      <tr style="background:#F5F7FA;"><td style="padding:6px 8px;color:#64748B;font-weight:600;">Contract</td>
          <td style="padding:6px 8px;color:#1a1a1a;">{contract}</td></tr>
      <tr><td style="padding:6px 8px;color:#64748B;font-weight:600;">Tenure</td>
          <td style="padding:6px 8px;color:#1a1a1a;">{tenure} months</td></tr>
    </table>
    """, unsafe_allow_html=True)

with right:
    st.markdown('<p class="section-hdr">Customer Segment</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="strategy-card">
      <div class="seg-title">{seg_label}</div>
      <div class="seg-action">{seg_action}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── SHAP waterfall ────────────────────────────────────────────────────────────
shap_col, interp_col = st.columns([1.6, 1])

with shap_col:
    st.markdown('<p class="section-hdr">Top Churn Drivers — SHAP Explanation</p>', unsafe_allow_html=True)
    st.caption("Red bars push toward churn · Blue bars push away from churn · Length = strength of influence")

    # Build sorted top-N feature contributions
    N = 12
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:N]
    top_names  = [all_feat_names[i] for i in sorted_idx]
    top_vals   = [shap_vals[i] for i in sorted_idx]
    top_data   = [X_proc_df.iloc[0][all_feat_names[i]] for i in sorted_idx]

    # Reverse for horizontal bar (bottom = most important)
    top_names  = top_names[::-1]
    top_vals   = top_vals[::-1]
    colors     = ["#C0392B" if v > 0 else "#0E7B8C" for v in top_vals]

    plt.style.use("default")
    fig_shap, ax2 = plt.subplots(figsize=(7, 4.5), facecolor="white")
    ax2.set_facecolor("white")
    bars = ax2.barh(range(N), top_vals, color=colors, height=0.65, edgecolor="white")

    # Clean feature name labels
    def clean_name(n):
        n = n.replace("cat__", "").replace("num__", "")
        if "_" in n:
            parts = n.split("_", 1)
            return f"{parts[0].replace('_',' ')}: {parts[1].replace('_',' ')}"
        return n.replace("_", " ")

    ax2.set_yticks(range(N))
    ax2.set_yticklabels([clean_name(n) for n in top_names], fontsize=9.5)
    ax2.axvline(0, color="#94A3B8", lw=0.8)
    ax2.set_xlabel("SHAP Value (impact on churn probability)", fontsize=9, color="#64748B")
    ax2.spines[["top","right","left"]].set_visible(False)
    ax2.tick_params(left=False, colors="#374151")
    ax2.xaxis.label.set_color("#64748B")
    ax2.tick_params(axis="x", colors="#94A3B8", labelsize=8)
    fig_shap.tight_layout(pad=0.5)
    st.pyplot(fig_shap, use_container_width=True)
    plt.close()

with interp_col:
    st.markdown('<p class="section-hdr">Interpretation Guide</p>', unsafe_allow_html=True)

    # Top 3 push factors and top 3 protective factors
    push  = [(all_feat_names[i], shap_vals[i]) for i in np.argsort(shap_vals)[::-1][:3] if shap_vals[i] > 0]
    guard = [(all_feat_names[i], shap_vals[i]) for i in np.argsort(shap_vals)[:3]        if shap_vals[i] < 0]

    if push:
        rows = "".join(f'<div style="padding:4px 0;color:#1a1a1a;font-size:0.88rem;">🔴 <b>{feat.replace("_"," ")}</b> &nbsp;<span style="color:#C0392B;">(+{val:.3f})</span></div>' for feat, val in push)
        st.markdown(f'<div style="margin-bottom:0.5rem;"><div style="font-weight:700;color:#1a1a1a;margin-bottom:4px;">Pushing toward churn:</div>{rows}</div>', unsafe_allow_html=True)

    if guard:
        rows = "".join(f'<div style="padding:4px 0;color:#1a1a1a;font-size:0.88rem;">🟢 <b>{feat.replace("_"," ")}</b> &nbsp;<span style="color:#1A7A4A;">({val:.3f})</span></div>' for feat, val in guard)
        st.markdown(f'<div style="margin-bottom:0.8rem;"><div style="font-weight:700;color:#1a1a1a;margin-bottom:4px;">Protecting against churn:</div>{rows}</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:1px solid #e2e8f0;margin:0.6rem 0;">', unsafe_allow_html=True)
    st.markdown("""<div style="font-size:0.85rem;color:#374151;">
    <b>What SHAP values mean:</b><br>
    • Each value is the additive contribution of that feature to the final score<br>
    • Baseline = average prediction across all customers<br>
    • Sum of all SHAP values + baseline = final probability<br>
    • Larger magnitude = stronger influence
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="warn-box">
    ⚠️ <b>Note for users:</b> This score is a probability estimate, not a guarantee. 
    Always pair model output with agent judgment before contacting a customer.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small style='color:#94A3B8'>Model: Calibrated GBT + SMOTE-ENN · "
    "Tuned via Optuna (40 trials) · Threshold: 0.38 · "
    "Train ROC-AUC: 0.849 · DATA 4382 Capstone II · UTA</small>",
    unsafe_allow_html=True
)
