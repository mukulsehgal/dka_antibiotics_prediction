#!/usr/bin/env python3
import re
import io
import gzip
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import requests

st.set_page_config(page_title="DKA Antibiotics Predictor", layout="wide")

# -----------------------------
# 1) Load trained model (from secret URL or local file)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    url = st.secrets.get("MODEL_URL", "").strip()
    if url:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        buf = io.BytesIO(r.content)
        # try gzipped first
        try:
            with gzip.GzipFile(fileobj=buf) as gz:
                return joblib.load(gz)
        except OSError:
            buf.seek(0)
            return joblib.load(buf)

    # local fallbacks
    try:
        return joblib.load("rf_antibiotics_model.pkl")
    except Exception:
        pass
    try:
        with gzip.open("rf_antibiotics_model.pkl.gz", "rb") as f:
            return joblib.load(f)
    except Exception as e:
        raise RuntimeError(
            "Model not found. Place rf_antibiotics_model.pkl(.gz) in repo or set MODEL_URL in secrets."
        ) from e

model = load_model()

# -----------------------------
# 2) Inspect model to learn expected features
# -----------------------------
def get_required_feature_sets(pipeline):
    pre = pipeline.named_steps.get("pre") or pipeline.named_steps.get("preprocessor")
    if pre is None:
        exp_all = list(getattr(pipeline, "feature_names_in_", []))
        return exp_all, [], exp_all

    exp_num, exp_cat = [], []
    for name, transformer, cols in pre.transformers_:
        if name == "num":
            exp_num.extend(list(cols))
        elif name == "cat":
            exp_cat.extend(list(cols))
    # unique, preserve order
    exp_num = list(dict.fromkeys(exp_num))
    exp_cat = list(dict.fromkeys(exp_cat))
    return exp_num, exp_cat, exp_num + exp_cat

EXPECTED_NUM, EXPECTED_CAT, EXPECTED_ALL = get_required_feature_sets(model)

# -----------------------------
# 3) Header normalization & synonyms
# -----------------------------
def norm(s: str) -> str:
    s = str(s).replace("\xa0", " ")
    s = re.sub(r"[ _]+", " ", s)
    return s.strip().upper()

SYNONYMS = {
    # gases/electrolytes
    "PHVEN":        ["PH", "PH VENOUS", "VENOUS PH", "PH_VENOUS"],
    "PCO2VEN":      ["PCO2", "PCO2 VBG", "PCO2 VENOUS", "PCO2_VENOUS"],
    "HCO3VEN":      ["HCO3", "BICARB", "BICARBONATE", "HCO3 VBG", "HCO3 VENOUS"],
    "SODIUM":       ["NA", "NA+"],
    "POTASSIUM":    ["K", "K+"],
    # heme/chem
    "WBC":          ["WBC COUNT", "WHITE BLOOD CELL COUNT"],
    "PLT":          ["PLATELET", "PLATELETS"],
    "HGB":          ["HEMOGLOBIN"],
    "RDW":          ["RED CELL DISTRIBUTION WIDTH"],
    "MCV":          ["MEAN CORPUSCULAR VOLUME"],
    "MPV":          ["MEAN PLATELET VOLUME"],
    "CREATININE":   ["CRE"],
    "BUN":          ["UREA", "UREA NITROGEN"],
    "LIPASE":       ["SERUM LIPASE"],
    "HBA1C":        ["HB A1C", "HEMOGLOBIN A1C", "HBA1C%"],
    "ANIONGAP":     ["ANION GAP", "AGAP"],
    # inflammatory
    "CRP":          ["C REACTIVE PROTEIN", "C-REACTIVE PROTEIN"],
    "PCAL":         ["PROCALCITONIN"],
    # demographics
    "AGE AT VISIT": ["AGE", "AGE (YEARS)", "AGE_AT_VISIT"],
    "GENDER":       ["SEX"],
    "RACE":         ["RACE/ETHNICITY", "RACE ETHNICITY"],
    "ETHNICITY":    ["ETHNIC GROUP"],
}

REV = {}
for canonical, alts in SYNONYMS.items():
    REV[norm(canonical)] = canonical
    for a in alts:
        REV[norm(a)] = canonical

# -----------------------------
# 4) Column mapping / coercion
# -----------------------------
def map_and_coerce(df: pd.DataFrame):
    # uploaded -> canonical
    mapped = {}
    norm_expected = {norm(x): x for x in EXPECTED_ALL}
    for col in df.columns:
        nc = norm(col)
        if nc in REV:
            mapped[col] = REV[nc]
        elif nc in norm_expected:
            mapped[col] = norm_expected[nc]

    df2 = df.rename(columns=mapped).copy()

    # Create any missing expected columns
    missing = [c for c in EXPECTED_ALL if c not in df2.columns]
    for c in missing:
        if c in EXPECTED_CAT:
            df2[c] = "Unknown"
        else:
            df2[c] = np.nan  # pipeline median imputer handles numerics

    # Reorder to training order
    df2 = df2.reindex(columns=EXPECTED_ALL)

    # Coerce dtypes
    for c in EXPECTED_CAT:
        df2[c] = df2[c].astype(str)
    for c in EXPECTED_NUM:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

    return df2, mapped, missing

# -----------------------------
# 5) Template generator
# -----------------------------
def make_template_df(include_examples: bool = True, use_aliases: bool = False) -> pd.DataFrame:
    """
    Build a template with expected columns (or common aliases if use_aliases=True).
    """
    cols = []
    if use_aliases:
        # Prefer a friendly alias if available; else the canonical name
        alias_for = {canon: (SYNONYMS[canon][0] if SYNONYMS.get(canon) else canon) for canon in EXPECTED_ALL}
        cols = [alias_for[c] for c in EXPECTED_ALL]
    else:
        cols = list(EXPECTED_ALL)

    data = {}
    for i, c in enumerate(cols):
        # keep categorical as 'Unknown', numeric as NaN
        canonical = EXPECTED_ALL[i]  # positional match
        if canonical in EXPECTED_CAT:
            data[c] = ["Unknown"] if include_examples else []
        else:
            data[c] = [np.nan] if include_examples else []
    return pd.DataFrame(data)

# -----------------------------
# 6) UI
# -----------------------------
st.title("Antibiotics Prediction for DKA (Bulk Upload)")
st.warning("**Upload de-identified data only (no PHI).** The app auto-maps column names and backfills missing features.")

# Template downloads
c1, c2, c3 = st.columns([1,1,2])
with c1:
    include_examples = st.toggle("Template: 1 example row", value=True)
with c2:
    use_aliases = st.toggle("Template with common aliases", value=False)

template_df = make_template_df(include_examples=include_examples, use_aliases=use_aliases)
st.download_button(
    "⬇️ Download template CSV",
    data=template_df.to_csv(index=False).encode("utf-8"),
    file_name="DKA_antibiotics_template.csv",
    mime="text/csv",
)

st.divider()

file = st.file_uploader("Upload a file to score (CSV / Excel)", type=["csv", "xlsx", "xls"])

if file is not None:
    # Read file
    if file.name.lower().endswith(".csv"):
        raw = pd.read_csv(file)
    else:
        raw = pd.read_excel(file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(raw.head(), use_container_width=True)

    # Map & coerce
    X, mapped, missing = map_and_coerce(raw)

    with st.expander("Column Mapping"):
        if mapped:
            st.write("**Mapped columns (uploaded → model feature)**")
            st.dataframe(pd.DataFrame({"UPLOADED": list(mapped.keys()), "MAPPED_TO": list(mapped.values())}),
                         use_container_width=True)
        else:
            st.info("No header synonyms were matched. If your headers already match training names, that’s fine.")
        if missing:
            st.warning(f"Missing features auto-filled: {missing}")

    with st.expander("Processed feature dtypes (what the model sees)"):
        st.write(X.dtypes)

    # Predict
    try:
        proba = model.predict_proba(X)[:, 1]
        pred = model.predict(X)

        out = raw.copy()
        out["ANTIBIOTICS_PROB_%"] = (proba * 100).round(2)
        out["ANTIBIOTICS_PRED"] = pred  # 1=Yes, 0=No

        st.subheader("Predictions (first 20 rows)")
        st.dataframe(out.head(20), use_container_width=True)

        st.download_button(
            "⬇️ Download predictions (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="antibiotics_predictions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error("Prediction failed. Check mapping and missing features above.")
        st.exception(e)

with st.expander("Model Info"):
    st.write("**Expected numeric features:**", EXPECTED_NUM)
    st.write("**Expected categorical features:**", EXPECTED_CAT)

