import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
import pickle
from preprocess import preprocess
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Sentiment Analysis PLN Mobile",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# PALET WARNA — konsisten di seluruh app
# ─────────────────────────────────────────
C_NAVY    = "#003f88"
C_BLUE    = "#0066cc"
C_ACCENT  = "#f5a623"
C_RED     = "#c0392b"
C_BG      = "#f4f6fb"
C_SURFACE = "#ffffff"
C_MUTED   = "#6b7a99"
C_TEXT    = "#1a2340"
C_GREEN   = "#27ae60"

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');

/* Global */
html, body, [class*="css"], * {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #f4f6fb;
}

[data-testid="stHeader"] {
    background: transparent;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #003f88 !important;
}
[data-testid="stSidebar"] section {
    padding-top: 1.5rem;
}
[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.8) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
    color: rgba(255,255,255,0.5) !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: rgba(255,255,255,0.6) !important;
    fill: rgba(255,255,255,0.6) !important;
}
[data-testid="stSidebar"] [data-baseweb="menu"] {
    background-color: #003f88 !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}
[data-testid="stSidebar"] [role="option"] {
    color: rgba(255,255,255,0.8) !important;
}
[data-testid="stSidebar"] [role="option"]:hover {
    background-color: rgba(255,255,255,0.1) !important;
}
[data-testid="stSidebarNav"] {
    display: none;
}

/* ── Judul sidebar ── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 1rem 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 1.5rem;
}
.sidebar-icon {
    width: 40px; height: 40px;
    background: #f5a623;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 600; font-size: 13px; color: #1a1000;
    flex-shrink: 0;
}
.sidebar-label { line-height: 1.3; }
.sidebar-title { font-size: 14px; font-weight: 600; color: #ffffff; }
.sidebar-sub   { font-size: 11px; color: rgba(255,255,255,0.5); }

/* ── Page title ── */
h1 {
    color: #003f88 !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    border-bottom: 3px solid #f5a623;
    padding-bottom: 10px;
    margin-bottom: 1.5rem !important;
}
h2 {
    color: #1a2340 !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    margin-top: 1.8rem !important;
    margin-bottom: 0.75rem !important;
}
h3 {
    color: #6b7a99 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.5rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #003f88 !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 10px 24px !important;
    transition: background 0.15s ease !important;
}
.stButton > button:hover {
    background-color: #0066cc !important;
    color: #ffffff !important;
}
.stButton > button:active {
    background-color: #002d63 !important;
}

/* ── Text area ── */
.stTextArea textarea {
    background-color: #ffffff !important;
    border: 1px solid rgba(0,63,136,0.15) !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    color: #1a2340 !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: #0066cc !important;
    box-shadow: 0 0 0 3px rgba(0,102,204,0.1) !important;
}
.stTextArea label {
    color: #6b7a99 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ── Alert boxes ── */
.stSuccess {
    background-color: #eaf6ef !important;
    color: #1a7a44 !important;
    border-left: 4px solid #27ae60 !important;
    border-radius: 0 8px 8px 0 !important;
}
.stWarning {
    background-color: #fff8e6 !important;
    color: #7a5800 !important;
    border-left: 4px solid #f5a623 !important;
    border-radius: 0 8px 8px 0 !important;
}
.stError {
    background-color: #fdecea !important;
    color: #a82828 !important;
    border-left: 4px solid #c0392b !important;
    border-radius: 0 8px 8px 0 !important;
}
.stInfo {
    background-color: #eef3fc !important;
    color: #003f88 !important;
    border-left: 4px solid #003f88 !important;
    border-radius: 0 8px 8px 0 !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid rgba(0,63,136,0.1) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrameResizable"] {
    border-radius: 10px !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid rgba(0,63,136,0.12) !important;
    border-radius: 12px !important;
    padding: 18px 20px !important;
}
[data-testid="stMetricLabel"] {
    color: #6b7a99 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    color: #1a2340 !important;
    font-size: 26px !important;
    font-weight: 600 !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #003f88 !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(0,63,136,0.1) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR BRAND
# ─────────────────────────────────────────
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="sidebar-icon">PLN</div>
    <div class="sidebar-label">
        <div class="sidebar-title">PLN Mobile</div>
        <div class="sidebar-sub">Sentiment Dashboard</div>
    </div>
</div>
""", unsafe_allow_html=True)

menu = ["Prediksi Teks", "Visualisasi Data Mentah", "Visualisasi Data Preprocessing"]
choice = st.sidebar.selectbox("Menu", menu)

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model_nb_pln.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer_pln.pkl")


with open(model_path, "rb") as f:
    nb = pickle.load(f)
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

def predict_text(text):
    text_proc = preprocess(text)
    vec = vectorizer.transform([text_proc])
    return nb.predict(vec)[0]

# ─────────────────────────────────────────
# HELPER: setup axes style
# ─────────────────────────────────────────
def style_ax(ax, fig):
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e0e4ed")
    ax.spines["bottom"].set_color("#e0e4ed")
    ax.tick_params(colors=C_MUTED, labelsize=11)
    ax.xaxis.label.set_color(C_MUTED)
    ax.yaxis.label.set_color(C_MUTED)
    ax.title.set_color(C_TEXT)
    ax.grid(axis="y", color="#e0e4ed", linestyle="--", alpha=0.7, linewidth=0.8)

# ═════════════════════════════════════════
# MENU 1 — PREDIKSI TEKS
# ═════════════════════════════════════════
if choice == "Prediksi Teks":
    st.title("Prediksi Sentimen Ulasan")

    user_input = st.text_area(
        "Masukkan teks ulasan",
        placeholder="Contoh: Aplikasi PLN Mobile sangat membantu, tidak perlu antri lagi...",
        height=130
    )

    if st.button("Prediksi Sekarang"):
        if user_input.strip() == "":
            st.warning("Teks tidak boleh kosong.")
        else:
            with st.spinner("Menganalisis teks..."):
                hasil = predict_text(user_input)

            if hasil == "positive":
                st.success(f"✅  Prediksi sentimen: **POSITIF**")
            else:
                st.error(f"❌  Prediksi sentimen: **NEGATIF**")

# ═════════════════════════════════════════
# MENU 2 — VISUALISASI DATA MENTAH
# ═════════════════════════════════════════
elif choice == "Visualisasi Data Mentah":
    st.title("Visualisasi Data Mentah")

    raw_path = "ulasan_pln_mobile.csv"
    df_raw = pd.read_csv(raw_path)

    # Metric row
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Ulasan", f"{len(df_raw):,}")
    col2.metric("Rata-rata Rating", f"{df_raw['score'].mean():.2f} ★")
    col3.metric("Kolom Data", len(df_raw.columns))

    st.markdown("---")

    # Preview tabel
    st.subheader("Preview Data")
    st.dataframe(df_raw.head(10), height=220, use_container_width=True)

    st.markdown("---")

    # ── Chart 1: Distribusi rating awal
    st.subheader("Distribusi Rating Awal")
    fig, ax = plt.subplots(figsize=(7, 4))
    style_ax(ax, fig)

    counts = df_raw["score"].value_counts().sort_index()
    bar_colors = [C_RED, C_RED, "#a3bef7", "#5b8ef7", C_NAVY]
    bars = ax.bar(counts.index, counts.values,
                  color=bar_colors[:len(counts)],
                  edgecolor="none", width=0.6)
    ax.set_xlabel("Rating", labelpad=8)
    ax.set_ylabel("Jumlah Ulasan", labelpad=8)
    ax.set_title("Distribusi Rating Awal", fontsize=13, fontweight="500", pad=12)
    ax.bar_label(bars, fmt="%d", fontsize=10, color=C_MUTED, padding=4)

    legend_patches = [
        mpatches.Patch(color=C_RED,    label="Negatif (★1–2)"),
        mpatches.Patch(color="#a3bef7", label="Netral (★3–4)"),
        mpatches.Patch(color=C_NAVY,   label="Positif (★5)"),
    ]
    ax.legend(handles=legend_patches, fontsize=10, frameon=False,
              labelcolor=C_MUTED)
    st.pyplot(fig)

    st.markdown("---")

    # Label filtering & balancing
    df_raw["label"] = df_raw["score"].apply(
        lambda x: "positive" if x == 5 else ("negative" if x <= 2 else None)
    )
    df_filtered = df_raw.dropna(subset=["label"])

    col_a, col_b = st.columns(2)

    # ── Chart 2: Setelah filter
    with col_a:
        st.subheader("Setelah Filter Rating")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        style_ax(ax2, fig2)
        counts2 = df_filtered["label"].value_counts()
        bar2 = ax2.bar(
            counts2.index, counts2.values,
            color=[C_NAVY if l == "positive" else C_RED for l in counts2.index],
            edgecolor="none", width=0.5
        )
        ax2.set_xlabel("Label", labelpad=8)
        ax2.set_ylabel("Jumlah", labelpad=8)
        ax2.set_title("Filter Rating 1–2 & 5", fontsize=12, fontweight="500", pad=10)
        ax2.bar_label(bar2, fmt="%d", fontsize=10, color=C_MUTED, padding=4)
        st.pyplot(fig2)

    # ── Chart 3: Setelah balancing
    df_pos = df_filtered[df_filtered["label"] == "positive"]
    df_neg = df_filtered[df_filtered["label"] == "negative"]
    n = min(len(df_pos), len(df_neg))
    df_balanced = pd.concat([
        df_pos.sample(n, random_state=42),
        df_neg.sample(n, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    with col_b:
        st.subheader("Setelah Balancing")
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        style_ax(ax3, fig3)
        counts3 = df_balanced["label"].value_counts()
        bar3 = ax3.bar(
            counts3.index, counts3.values,
            color=[C_NAVY if l == "positive" else C_RED for l in counts3.index],
            edgecolor="none", width=0.5
        )
        ax3.set_xlabel("Label", labelpad=8)
        ax3.set_ylabel("Jumlah", labelpad=8)
        ax3.set_title(f"Balanced ({n} per label)", fontsize=12, fontweight="500", pad=10)
        ax3.bar_label(bar3, fmt="%d", fontsize=10, color=C_MUTED, padding=4)
        st.pyplot(fig3)

# ═════════════════════════════════════════
# MENU 3 — VISUALISASI PREPROCESSING
# ═════════════════════════════════════════
elif choice == "Visualisasi Data Preprocessing":
    st.title("Visualisasi Data Preprocessing")

    pre_path = "preprocessing_train.csv"
    try:
        df_pre = pd.read_csv(pre_path)
        df_pre["text_final"] = df_pre["text_final"].fillna("").astype(str)

        # Metric row
        total = len(df_pre)
        n_pos = (df_pre["label"] == "positive").sum()
        n_neg = (df_pre["label"] == "negative").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Data", f"{total:,}")
        col2.metric("Positif", f"{n_pos:,}")
        col3.metric("Negatif", f"{n_neg:,}")

        st.markdown("---")

        # Preview tabel
        st.subheader("Preview Data Setelah Preprocessing")
        st.dataframe(df_pre.head(10), height=220, use_container_width=True)

        st.markdown("---")

        # ── Chart: Distribusi label
        st.subheader("Distribusi Label")
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        style_ax(ax4, fig4)
        counts4 = df_pre["label"].value_counts()
        bar4 = ax4.bar(
            counts4.index, counts4.values,
            color=[C_NAVY if l == "positive" else C_RED for l in counts4.index],
            edgecolor="none", width=0.5
        )
        ax4.set_xlabel("Label", labelpad=8)
        ax4.set_ylabel("Jumlah", labelpad=8)
        ax4.set_title("Distribusi Label Preprocessing", fontsize=12, fontweight="500", pad=10)
        ax4.bar_label(bar4, fmt="%d", fontsize=10, color=C_MUTED, padding=4)
        st.pyplot(fig4)

        st.markdown("---")

        # ── WordClouds
        col_wc1, col_wc2 = st.columns(2)

        with col_wc1:
            text_pos = " ".join(df_pre[df_pre["label"] == "positive"]["text_final"])
            if text_pos.strip():
                st.subheader("WordCloud — Positif")
                wc_pos = WordCloud(
                    width=700, height=350,
                    background_color=C_SURFACE,
                    colormap="Blues",
                    max_words=80,
                    prefer_horizontal=0.9,
                    contour_color=C_NAVY,
                    contour_width=1,
                ).generate(text_pos)
                fig_wc1, ax_wc1 = plt.subplots(figsize=(6, 3))
                fig_wc1.patch.set_facecolor(C_SURFACE)
                ax_wc1.imshow(wc_pos, interpolation="bilinear")
                ax_wc1.axis("off")
                st.pyplot(fig_wc1)

        with col_wc2:
            text_neg = " ".join(df_pre[df_pre["label"] == "negative"]["text_final"])
            if text_neg.strip():
                st.subheader("WordCloud — Negatif")
                wc_neg = WordCloud(
                    width=700, height=350,
                    background_color=C_SURFACE,
                    colormap="Reds",
                    max_words=80,
                    prefer_horizontal=0.9,
                    contour_color=C_RED,
                    contour_width=1,
                ).generate(text_neg)
                fig_wc2, ax_wc2 = plt.subplots(figsize=(6, 3))
                fig_wc2.patch.set_facecolor(C_SURFACE)
                ax_wc2.imshow(wc_neg, interpolation="bilinear")
                ax_wc2.axis("off")
                st.pyplot(fig_wc2)

        st.markdown("---")

        # ── Confusion Matrix
        st.subheader("Confusion Matrix (Sample 200 Data)")
        sample_df = df_pre.sample(200, random_state=42)
        y_true = sample_df["label"]

        with st.spinner("Menghitung prediksi model..."):
            y_pred = [predict_text(t) for t in sample_df["text_final"]]

        cm = confusion_matrix(y_true, y_pred, labels=["positive", "negative"])

        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        fig_cm.patch.set_facecolor(C_BG)
        ax_cm.set_facecolor(C_BG)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Positif", "Negatif"],
            yticklabels=["Positif", "Negatif"],
            ax=ax_cm,
            linewidths=2,
            linecolor=C_BG,
            annot_kws={"size": 16, "weight": "600", "color": C_TEXT},
            cbar=False,
        )
        ax_cm.set_xlabel("Prediksi", labelpad=10, color=C_MUTED, fontsize=12)
        ax_cm.set_ylabel("Aktual", labelpad=10, color=C_MUTED, fontsize=12)
        ax_cm.set_title("Confusion Matrix", fontsize=13, fontweight="500",
                         color=C_TEXT, pad=12)
        ax_cm.tick_params(colors=C_MUTED, labelsize=11)
        st.pyplot(fig_cm)

    except FileNotFoundError:
        st.warning("File `hasil_preprocessing.csv` belum ditemukan. Jalankan preprocessing terlebih dahulu.")
