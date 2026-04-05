# Copyright (c) 2026 Komal Dahiya
# GitHub: https://github.com/komaldahiya912/brain-tumour-grading
# Licensed under CC BY-NC 4.0 — Non-commercial use only.
# Credit must be given if this code is used or adapted.

"""
app.py — Brain Tumour Grading System v2.0
Dual Pipeline: MRI (Grade 1/2) + Clinical features (LGG/GBM)
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, io, math
from datetime import datetime
import pandas as pd
from skimage.transform import resize

from model_loader import BrainTumorPredictor, VQC2Predictor
from database    import PredictionDatabase

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Image as RLImage, Table, TableStyle)
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    _PDF = True
except ImportError:
    _PDF = False


# ════════════════════════════════════════════════════════════════════
#  CACHED LOADERS
# ════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_p1():
    return BrainTumorPredictor()

@st.cache_resource
def load_p2():
    return VQC2Predictor()

@st.cache_resource
def load_db():
    return PredictionDatabase()


# ════════════════════════════════════════════════════════════════════
#  PATIENT INFO FORM  (shared by both modes)
# ════════════════════════════════════════════════════════════════════
def patient_info_form(key_prefix=""):
    """Renders patient detail fields. Returns dict."""
    st.subheader("Patient Details")
    c1, c2 = st.columns(2)
    with c1:
        name       = st.text_input("Full Name *",       placeholder="e.g. Ramesh Kumar",        key=f"{key_prefix}_name")
        patient_id = st.text_input("Patient ID",        placeholder="e.g. AIIMS-2024-001",       key=f"{key_prefix}_pid")
        phone      = st.text_input("Phone Number",      placeholder="e.g. 9876543210",           key=f"{key_prefix}_phone")
    with c2:
        gender     = st.selectbox("Gender",             ["", "Male", "Female", "Other"],         key=f"{key_prefix}_gender")
        dob        = st.date_input("Date of Birth",     value=None, min_value=datetime(1900,1,1).date(),
                                   max_value=datetime.today().date(),                             key=f"{key_prefix}_dob")
    return {
        "name":       name,
        "patient_id": patient_id,
        "phone":      phone,
        "gender":     gender,
        "dob":        str(dob) if dob else "",
    }


# ════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════
def make_overlay(orig_arr, mask, alpha=0.45):
    if orig_arr.shape != mask.shape:
        mask = resize(mask, orig_arr.shape, preserve_range=True, anti_aliasing=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(orig_arr, cmap="gray", vmin=0, vmax=255)
    ov = np.zeros((*mask.shape, 4))
    ov[mask > 0.5, 0] = 1.0
    ov[mask > 0.5, 3] = alpha
    ax.imshow(ov, interpolation="nearest")
    patch = mpatches.Patch(color=(1,0,0,0.6), label="Tumour")
    ax.legend(handles=[patch], loc="lower right", fontsize=8)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor="white")
    buf.seek(0); plt.close()
    return buf


def prob_bar(lgg_p, gbm_p):
    fig, ax = plt.subplots(figsize=(4.5, 1.1))
    ax.barh(["LGG", "GBM"], [lgg_p, gbm_p],
            color=["#22C55E", "#EF4444"], height=0.55)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="grey", ls="--", lw=0.8)
    for i, v in enumerate([lgg_p, gbm_p]):
        ax.text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Probability", fontsize=8)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0); plt.close()
    return buf


def pdf_mri(pinfo, results, orig_arr, mask):
    if not _PDF: return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    ss  = getSampleStyleSheet()
    S   = []
    S.append(Paragraph("<b>Brain Tumour MRI Analysis Report</b>", ss["Title"]))
    S.append(Spacer(1,12))
    S.append(Paragraph(f"<b>Patient Name:</b> {pinfo['name']}", ss["Normal"]))
    S.append(Paragraph(f"<b>Patient ID:</b> {pinfo['patient_id']}", ss["Normal"]))
    S.append(Paragraph(f"<b>Gender:</b> {pinfo['gender']}  |  <b>DOB:</b> {pinfo['dob']}", ss["Normal"]))
    S.append(Paragraph(f"<b>Phone:</b> {pinfo['phone']}", ss["Normal"]))
    S.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", ss["Normal"]))
    S.append(Spacer(1,10))
    S.append(Paragraph("<b>Detection</b>", ss["Heading2"]))
    S.append(Paragraph(f"Tumour detected: {'Yes' if results['tumor_present'] else 'No'}", ss["Normal"]))
    S.append(Paragraph(f"Tumour area: {results['tumor_area']:.0f} pixels", ss["Normal"]))
    S.append(Spacer(1,8))
    S.append(Paragraph("<b>Grading</b>", ss["Heading2"]))
    S.append(Paragraph(f"Predicted grade: Grade {results['predicted_grade']}", ss["Normal"]))
    S.append(Paragraph(f"Confidence: {results['grade_confidence']:.4f}", ss["Normal"]))
    S.append(Spacer(1,8))
    S.append(Paragraph("<b>Segmentation Statistics</b>", ss["Heading2"]))
    for k, v in results["segmentation_stats"].items():
        S.append(Paragraph(f"  {k}: {v:.4f}", ss["Normal"]))
    S.append(Spacer(1,12))
    ov = make_overlay(orig_arr, mask)
    S.append(RLImage(ov, width=300, height=300))
    S.append(Spacer(1,12))
    S.append(Paragraph("<i>Not for clinical diagnosis.</i>", ss["Normal"]))
    doc.build(S); buf.seek(0)
    return buf


def pdf_clinical(pinfo, features, result):
    if not _PDF: return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    ss  = getSampleStyleSheet()
    S   = []
    S.append(Paragraph("<b>Brain Tumour Clinical Classification Report</b>", ss["Title"]))
    S.append(Spacer(1,12))
    S.append(Paragraph(f"<b>Patient Name:</b> {pinfo['name']}", ss["Normal"]))
    S.append(Paragraph(f"<b>Patient ID:</b> {pinfo['patient_id']}", ss["Normal"]))
    S.append(Paragraph(f"<b>Gender:</b> {pinfo['gender']}  |  <b>DOB:</b> {pinfo['dob']}", ss["Normal"]))
    S.append(Paragraph(f"<b>Phone:</b> {pinfo['phone']}", ss["Normal"]))
    S.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", ss["Normal"]))
    S.append(Spacer(1,10))
    S.append(Paragraph("<b>Input Features</b>", ss["Heading2"]))
    feat_rows = [
        ["Feature", "Value"],
        ["IDH1 Mutation",    "Mutated" if features["idh1"] else "Not Mutated"],
        ["Age at Diagnosis", f"{features['age']:.1f} years"],
        ["PTEN Mutation",    "Mutated" if features["pten"] else "Not Mutated"],
        ["EGFR Mutation",    "Mutated" if features["egfr"] else "Not Mutated"],
        ["ATRX Mutation",    "Mutated" if features["atrx"] else "Not Mutated"],
    ]
    t = Table(feat_rows, colWidths=[200, 200])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#1E3A5F")),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#F8FAFC"), colors.white]),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.grey),
        ("FONTSIZE",      (0,0),(-1,-1), 10),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
    ]))
    S.append(t); S.append(Spacer(1,10))
    S.append(Paragraph("<b>Result</b>", ss["Heading2"]))
    S.append(Paragraph(
        f"<b>Predicted class: {result['predicted_class']}</b>",
        ss["Normal"]
    ))
    S.append(Paragraph(f"Confidence: {result['confidence']*100:.1f}%", ss["Normal"]))
    S.append(Paragraph(f"LGG probability: {result['lgg_probability']*100:.1f}%", ss["Normal"]))
    S.append(Paragraph(f"GBM probability: {result['gbm_probability']*100:.1f}%", ss["Normal"]))
    S.append(Spacer(1,12))
    S.append(Paragraph("<i>Not for clinical diagnosis.</i>", ss["Normal"]))
    doc.build(S); buf.seek(0)
    return buf


# ════════════════════════════════════════════════════════════════════
#  PAGE: MODE 1 — MRI
# ════════════════════════════════════════════════════════════════════
def page_mode1(db):
    st.title("🧠 Mode 1 — MRI Scan Analysis")
    st.caption("ResNet50-UNet segmentation · 4-qubit VQC · LGG Grade 1 vs Grade 2")
    st.info("This mode grades LGG sub-types (Grade 1 vs Grade 2) from MRI images. For LGG vs GBM classification use Mode 2.")
    st.markdown("---")

    pinfo = patient_info_form(key_prefix="m1")

    st.markdown("---")
    st.subheader("Upload MRI Scan")
    uploaded = st.file_uploader(
        "Upload Brain MRI Scan *",
        type=["png", "jpg", "jpeg", "tiff", "tif"],
        help="Max 200MB. Greyscale or RGB brain MRI."
    )

    if not pinfo["name"]:
        st.warning("Enter patient name to proceed.")
        return
    if not uploaded:
        st.info("Upload an MRI scan to continue.")
        return

    os.makedirs("static/uploads", exist_ok=True)
    img_path = os.path.join("static", "uploads", uploaded.name)
    image_bytes = uploaded.getvalue()
    with open(img_path, "wb") as f:
        f.write(image_bytes)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original MRI")
        orig_img = Image.open(img_path).convert("L")
        st.image(orig_img, use_column_width=True)

    if not st.button("🔬 Analyse MRI", type="primary", use_container_width=True):
        return

    with st.spinner("Running segmentation + quantum grading …"):
        try:
            predictor = load_p1()
            results   = predictor.predict(img_path)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.exception(e)
            return

    orig_arr = np.array(orig_img)
    with col2:
        st.subheader("Tumour Overlay")
        st.image(make_overlay(orig_arr, results["tumor_mask"]), use_column_width=True)

    st.markdown("---")
    st.subheader("Results")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Tumour Detected",  "Yes ✓" if results["tumor_present"] else "No")
    m2.metric("Predicted Grade",  f"Grade {results['predicted_grade']}")
    m3.metric("Confidence",       f"{results['grade_confidence']:.3f}")
    m4.metric("Tumour Area",      f"{results['tumor_area']:.0f} px")

    with st.expander("Segmentation Details"):
        seg = results["segmentation_stats"]
        s1,s2,s3,s4 = st.columns(4)
        s1.metric("Mean Prob",    f"{seg['mean_prob']:.4f}")
        s2.metric("Std Prob",     f"{seg['std_prob']:.4f}")
        s3.metric("Max Prob",     f"{seg['max_prob']:.4f}")
        s4.metric("Tumour Ratio", f"{seg['tumor_ratio']:.4f}")

    st.markdown("---")
    try:
        pid = db.save_prediction(pinfo, img_path, results, image_bytes)
        st.success(f"Saved to database (ID: {pid})")
    except Exception as e:
        st.error(f"Database save failed: {e}")

    if _PDF:
        buf = pdf_mri(pinfo, results, orig_arr, results["tumor_mask"])
        if buf:
            st.download_button(
                "⬇ Download PDF Report", data=buf.getvalue(),
                file_name=f"{pinfo['name'].replace(' ','_')}_MRI_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )

    st.warning("⚠ Research system only — not for clinical diagnosis.")


# ════════════════════════════════════════════════════════════════════
#  PAGE: MODE 2 — CLINICAL
# ════════════════════════════════════════════════════════════════════
def page_mode2(db):
    st.title("🧬 Mode 2 — Clinical Feature Classification")
    st.caption("VQC-2 Quantum Classifier · 5 qubits · 84.81% accuracy · 93.67% recall")
    st.info("Enter genomic mutation status from biopsy or TCGA report. Classifies LGG vs GBM.")
    st.markdown("---")

    pinfo = patient_info_form(key_prefix="m2")

    st.markdown("---")
    st.subheader("Mutation Data")
    c1, c2 = st.columns(2)
    with c1:
        idh1 = st.selectbox("IDH1 Mutation *", [0,1],
            format_func=lambda v: "Not Mutated (0)" if v==0 else "Mutated (1)",
            help="Mutated in ~80% of LGG vs ~5% of GBM", key="m2_idh1")
        pten = st.selectbox("PTEN Mutation *", [0,1],
            format_func=lambda v: "Not Mutated (0)" if v==0 else "Mutated (1)",
            help="PTEN loss common in GBM (~40%)", key="m2_pten")
        egfr = st.selectbox("EGFR Mutation *", [0,1],
            format_func=lambda v: "Not Mutated (0)" if v==0 else "Mutated (1)",
            help="EGFR amplification is a GBM hallmark", key="m2_egfr")
    with c2:
        age  = st.number_input("Age at Diagnosis * (years)",
            min_value=0.0, max_value=120.0, value=45.0, step=0.5,
            help="LGG median ~38 yrs · GBM median ~60 yrs", key="m2_age")
        atrx = st.selectbox("ATRX Mutation *", [0,1],
            format_func=lambda v: "Not Mutated (0)" if v==0 else "Mutated (1)",
            help="ATRX mutation is characteristic of LGG astrocytoma", key="m2_atrx")

    st.markdown("---")
    if not pinfo["name"]:
        st.warning("Enter patient name to proceed.")
        return

    st.markdown("---")
    threshold = st.slider(
        "🎚 GBM Detection Sensitivity (lower = fewer missed cancers, more false alarms)",
        min_value=0.05, max_value=0.50, value=0.30, step=0.05,
        help="Default 0.30 recommended for clinical use. Lower values reduce missed GBM but increase false positives."
    )
    st.caption(f"At threshold {threshold:.2f} — model flags GBM if GBM probability > {threshold*100:.0f}%")

    if not st.button("⚛ Classify Tumour", type="primary", use_container_width=True):
        return

    with st.spinner("Running quantum classification …"):
        try:
            predictor = load_p2()
            result    = predictor.predict(idh1, age, pten, egfr, atrx, threshold)
        except Exception as e:
            st.error(f"Classification failed: {e}")
            st.exception(e)
            return

    st.markdown("---")
    cls  = result["predicted_class"]
    conf = result["confidence"]

    if cls == "GBM":
        st.markdown(
            f'<div style="background:#FEF2F2;border:2px solid #EF4444;'
            f'border-radius:10px;padding:20px;text-align:center;">'
            f'<div style="font-size:1.8rem;font-weight:700;color:#DC2626">🔴 GBM Detected</div>'
            f'<div style="color:#555;margin-top:4px">Glioblastoma Multiforme — Grade IV</div>'
            f'<div style="font-size:1.2rem;margin-top:10px">Confidence: <b>{conf*100:.1f}%</b></div>'
            f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div style="background:#DCFCE7;border:2px solid #22C55E;'
            f'border-radius:10px;padding:20px;text-align:center;">'
            f'<div style="font-size:1.8rem;font-weight:700;color:#15803D">🟢 LGG Detected</div>'
            f'<div style="color:#555;margin-top:4px">Low Grade Glioma</div>'
            f'<div style="font-size:1.2rem;margin-top:10px">Confidence: <b>{conf*100:.1f}%</b></div>'
            f'</div>', unsafe_allow_html=True)

    if 0.45 < conf < 0.65:
        st.warning("⚠ Confidence near 50% — result is uncertain. Consider additional testing.")

    st.markdown("")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Prediction",      cls)
    m2.metric("Confidence",      f"{conf*100:.1f}%")
    m3.metric("LGG Probability", f"{result['lgg_probability']*100:.1f}%")
    m4.metric("GBM Probability", f"{result['gbm_probability']*100:.1f}%")
    st.image(prob_bar(result["lgg_probability"], result["gbm_probability"]), width=420)

    with st.expander("🔬 Feature Interpretation"):
        interp = {
            "IDH1":  ("Mutated — strongly favours LGG" if idh1 else "Not Mutated — strongly favours GBM"),
            "Age":   (f"{age:.1f} yrs — younger, associated with LGG" if age < 45 else f"{age:.1f} yrs — older, associated with GBM"),
            "PTEN":  ("Mutated — favours GBM" if pten else "Not Mutated — no strong signal"),
            "EGFR":  ("Mutated — favours GBM" if egfr else "Not Mutated — favours LGG"),
            "ATRX":  ("Mutated — favours LGG astrocytoma" if atrx else "Not Mutated — neutral to GBM signal"),
        }
        for feat, txt in interp.items():
            st.markdown(f"**{feat}:** {txt}")

    features_dict = {"idh1":idh1,"age":age,"pten":pten,"egfr":egfr,"atrx":atrx}
    try:
        pid = db.save_clinical_prediction(pinfo, features_dict, result)
        st.success(f"Saved to database (ID: {pid})")
    except Exception as e:
        st.error(f"Database save failed: {e}")

    if _PDF:
        buf = pdf_clinical(pinfo, features_dict, result)
        if buf:
            st.download_button(
                "⬇ Download PDF Report", data=buf.getvalue(),
                file_name=f"{pinfo['name'].replace(' ','_')}_Clinical_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )

    st.warning("⚠ Research system only — not for clinical diagnosis.")


# ════════════════════════════════════════════════════════════════════
#  PAGE: HISTORY
# ════════════════════════════════════════════════════════════════════
def page_history(db):
    st.title("📋 Prediction History")

    # Search bar
    search = st.text_input("🔍 Search by patient name or patient ID", placeholder="Type to search…")
    st.markdown("---")

    tab1, tab2 = st.tabs(["🧠 MRI Predictions", "🧬 Clinical Predictions"])

    with tab1:
        preds = db.search_predictions(search) if search else db.get_all_predictions()
        if preds:
            df = pd.DataFrame([{
                "ID":p[0], "Name":p[1], "Patient ID":p[2],
                "Phone":p[3], "Gender":p[4], "DOB":p[5],
                "Date":p[6], "Tumour":"Yes" if p[8] else "No",
                "Grade":f"Grade {p[9]}", "Confidence":f"{p[10]:.3f}",
                "Area(px)":f"{p[11]:.0f}",
            } for p in preds])
            st.dataframe(df, hide_index=True, use_container_width=True)
            s1,s2,s3 = st.columns(3)
            tc = sum(1 for p in preds if p[8])
            s1.metric("Total Scans",      len(preds))
            s2.metric("Tumours Detected", tc)
            s3.metric("No Tumour",        len(preds)-tc)
        else:
            st.info("No MRI predictions found.")

    with tab2:
        cpreds = db.search_clinical_predictions(search) if search else db.get_all_clinical_predictions()
        if cpreds:
            df2 = pd.DataFrame([{
                "ID":c[0], "Name":c[1], "Patient ID":c[2],
                "Phone":c[3], "Gender":c[4], "DOB":c[5],
                "Date":c[6], "IDH1":c[7], "Age":f"{c[8]:.1f}",
                "PTEN":c[9], "EGFR":c[10], "ATRX":c[11],
                "Prediction":c[12], "Confidence":f"{c[13]*100:.1f}%",
            } for c in cpreds])
            st.dataframe(df2, hide_index=True, use_container_width=True)
            lgg_c = sum(1 for c in cpreds if c[12]=="LGG")
            gbm_c = sum(1 for c in cpreds if c[12]=="GBM")
            a1,a2,a3 = st.columns(3)
            a1.metric("Total Clinical", len(cpreds))
            a2.metric("LGG Predicted",  lgg_c)
            a3.metric("GBM Predicted",  gbm_c)
        else:
            st.info("No clinical predictions found.")


# ════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ════════════════════════════════════════════════════════════════════
def page_about():
    st.title("ℹ About This System")
    c1,c2 = st.columns([3,2])
    with c1:
        st.markdown("""
## Brain Tumour Grading System v2.0
Dual-pipeline AI combining deep learning and quantum machine learning.

---

### Pipeline 1 — MRI Image Analysis
| | |
|---|---|
| Segmentation | ResNet50-UNet + Attention Gates |
| Input | Greyscale MRI (512×512) |
| Segmentation | Dice: 85.71% · IoU: 82.30% |
| Classifier | 4-qubit VQC, 3 layers, 24 parameters |
| Accuracy | ~54% |
| Task | LGG Grade 1 vs Grade 2 |
| Dataset | 110 LGG patients · 3,929 MRI scans |

> ⚠ LGG sub-grades differ at the molecular level, not visually.
> MRI texture alone cannot reliably separate Grade 1 from Grade 2.
> Use Mode 2 for stronger classification using genomic features.

---

### Pipeline 2 — Clinical Feature Classification
| | |
|---|---|
| Model | VQC-2 Variational Quantum Classifier |
| Qubits | 5 · Layers: 2 · Params: 20 |
| Feature map | RyRz+CZ |
| Ansatz | Ry+CNOT circular |
| **Accuracy** | **84.81%** |
| **Recall** | **93.67%** |
| Dataset | 862 TCGA patients · 5-fold CV |
| Features | IDH1, Age, PTEN, EGFR, ATRX |
| Task | LGG vs GBM |

---

### Why Quantum?
- Superposition: all 2⁵=32 basis states processed simultaneously
- CZ entanglement: models co-mutation patterns
- Exact analytical gradients via backpropagation
        """)
    with c2:
        st.markdown("""
### All 10 Models (Pipeline 2)

| Model | Accuracy | Recall |
|-------|----------|--------|
| Decision Tree | 86.08% | 90.63% |
| Random Forest | 85.96% | 90.63% |
| Logistic Reg. | 85.96% | 91.74% |
| XGBoost | 85.85% | 90.36% |
| SVC | 85.73% | 91.46% |
| KNN | 84.92% | 87.88% |
| **VQC-2 ★** | **84.81%** | **93.67% ★** |
| VQC-3 | 83.06% | 89.56% |
| VQC-1 | 82.96% | 88.19% |
| VQC-4 | 73.31% | 66.95% |

★ VQC-2 has the highest recall of all 10 models.

---

### Important Notice
❌ NOT for clinical diagnosis
❌ NOT FDA approved
❌ NOT a medical device

**Research and education only.**
        """)


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="Brain Tumour Grading",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""<style>
        .block-container{padding-top:1.4rem}
        [data-testid="stMetricValue"]{font-size:1.3rem}
    </style>""", unsafe_allow_html=True)

    db = load_db()

    st.sidebar.title("🧠 Brain Tumour Grading")
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox(
        "Navigate",
        ["🧠 Mode 1 — MRI Analysis",
         "🧬 Mode 2 — Clinical Features",
         "📋 Prediction History",
         "ℹ About"],
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Mode 1** — Upload MRI scan\n"
        "→ Segments & grades LGG\n\n"
        "**Mode 2** — Enter mutation data\n"
        "→ Classifies LGG vs GBM\n"
        "→ 84.81% acc · 93.67% recall"
    )
    st.sidebar.markdown("---")
    st.sidebar.warning("⚠ Research system — not for clinical use.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2026 Komal Dahiya  \nB.Tech AIDS Final Year Project")
    
    if   page == "🧠 Mode 1 — MRI Analysis":     page_mode1(db)
    elif page == "🧬 Mode 2 — Clinical Features": page_mode2(db)
    elif page == "📋 Prediction History":          page_history(db)
    elif page == "ℹ About":                        page_about()


if __name__ == "__main__":
    main()
