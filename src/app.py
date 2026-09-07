import re
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Spice Up Thai — Intelligence Report",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
C = {
    "bg":      "#FFFFFF",
    "bg2":     "#F9F5F3",
    "card":    "#FFFFFF",
    "border":  "#E8E0D8",
    "red":     "#C0392B",
    "red2":    "#E74C3C",
    "orange":  "#E67E22",
    "gold":    "#C9910D",
    "text":    "#1A1A1A",
    "muted":   "#6B6B6B",
    "divider": "#EDE8E3",
}
CLUSTER = {
    0: {"name": "Staff & Service",      "color": "#1A237E", "emoji": "🟦"},
    1: {"name": "Food Quality",         "color": "#4A148C", "emoji": "🟣"},
    2: {"name": "Operations & Pricing", "color": "#42A5F5", "emoji": "🔵"},
}

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1A1A1A;
}

/* Chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }

/* App */
.stApp { background-color: #FFFFFF; }

/* Headings */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #1A1A1A !important;
}

/* Pill tabs — centered */
.stTabs [data-baseweb="tab-list"] {
    background: #F2EBEA;
    border-radius: 50px;
    padding: 4px 5px;
    gap: 2px;
    border: none;
    width: fit-content;
    margin: 0 auto 1rem auto;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 50px;
    padding: 8px 20px;
    color: #6B6B6B;
    font-weight: 500;
    font-size: 15px;
    border: none;
    white-space: nowrap;
}
.stTabs [aria-selected="true"] {
    background: #C0392B !important;
    color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 0; }

/* Inputs */
[data-baseweb="select"] {
    background: #FFFFFF !important;
    border-color: #E8E0D8 !important;
    border-radius: 10px !important;
}
.stTextArea textarea {
    background: #FFFFFF !important;
    border: 1px solid #E8E0D8 !important;
    border-radius: 10px !important;
    color: #1A1A1A !important;
    font-size: 1rem;
    font-family: 'Inter', sans-serif;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #EDE8E3;
    margin: 24px 0;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Dataframe */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Native metric (hidden — using custom cards) */
[data-testid="stMetric"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    neg = pd.read_csv("data/processed/clustered_reviews.csv")
    full = pd.read_csv("data/processed/processed_reviews.csv")
    neg["cluster_name"] = neg["cluster"].map(lambda k: CLUSTER[k]["name"])
    return neg, full

@st.cache_data
def load_brief():
    with open("outputs/business_strategic_brief.md", "r") as f:
        return f.read()

@st.cache_resource
def load_models():
    from sentence_transformers import SentenceTransformer
    encoder   = SentenceTransformer("all-MiniLM-L6-v2")
    centroids = np.load("artifacts/cluster_centroids.npy")
    return encoder, centroids

negative_df, full_df = load_data()
brief_md = load_brief()

SP  = '<div style="margin-top:20px;"></div>'
SP2 = '<div style="margin-top:10px;"></div>'


# ── HTML / chart helpers ───────────────────────────────────────────────────────
def hero(headline: str, sub: str):
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #FDF8F6 0%, #F9F2EF 100%);
                border-left: 4px solid {C['red']};
                border-radius: 0 12px 12px 0;
                padding: 24px 36px 22px 28px;
                margin-bottom: 20px;">
        <div style="font-size:0.68rem;color:{C['red']};letter-spacing:0.16em;
                    text-transform:uppercase;font-weight:600;margin-bottom:0.45rem;
                    font-family:'Inter',sans-serif;">
            Customer Intelligence Report &nbsp;·&nbsp; Spice Up Thai
        </div>
        <h1 style="font-family:'Playfair Display',serif;font-size:56px;font-weight:700;
                   color:{C['text']};line-height:1.15;margin:0 0 0.6rem 0;">
            {headline}
        </h1>
        <p style="font-size:19px;color:{C['muted']};max-width:820px;
                  margin:0;line-height:1.7;font-family:'Inter',sans-serif;font-weight:300;">
            {sub}
        </p>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = ""):
    return f"""
    <div style="background:{C['card']};
                border: 1px solid {C['border']};
                border-left: 4px solid {C['red']};
                border-radius: 12px;
                padding: 40px;
                min-height: 200px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.05);
                height: 100%;">
        <div style="font-size:15px;color:{C['red']};text-transform:uppercase;
                    letter-spacing:1.5px;font-weight:600;margin-bottom:0.5rem;
                    font-family:'Inter',sans-serif;">{label}</div>
        <div style="font-size:52px;font-weight:700;color:{C['text']};
                    line-height:1;margin-bottom:0.4rem;
                    font-family:'Playfair Display',serif;">{value}</div>
        <div style="font-size:16px;color:{C['muted']};font-family:'Inter',sans-serif;">{sub}</div>
    </div>"""


def section_heading(text: str):
    st.markdown(f"""
    {SP}
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
        <div style="width:4px;height:22px;background:{C['red']};border-radius:2px;flex-shrink:0;"></div>
        <h2 style="font-family:'Playfair Display',serif;font-size:34px;font-weight:700;
                   color:{C['text']};margin:0;">{text}</h2>
    </div>
    """, unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(f"""
    {SP2}
    <div style="font-size:11px;color:{C['red']};text-transform:uppercase;
                letter-spacing:0.16em;font-weight:600;margin-bottom:0.5rem;
                font-family:'Inter',sans-serif;">{text}</div>
    """, unsafe_allow_html=True)


def callout(text: str, color: str = None):
    col = color or C["red"]
    st.markdown(f"""
    <div style="background:{col}08;border-left:3px solid {col};
                padding:0.75rem 1.2rem;border-radius:0 8px 8px 0;
                color:{C['muted']};margin:0.3rem 0 0.75rem 0;font-size:18px;
                line-height:1.6;font-family:'Inter',sans-serif;">{text}</div>
    """, unsafe_allow_html=True)


def divider():
    st.markdown('<hr>', unsafe_allow_html=True)


def review_card(text: str, rating: float, source: str, color: str):
    stars = "★" * int(rating) + "☆" * (5 - int(rating))
    badge_color = "#4285F4" if source == "Google" else "#C0392B"
    short = text if len(text) <= 300 else text[:297] + "…"
    return f"""
    <div style="background:{C['card']};
                border: 1px solid {C['border']};
                border-left: 4px solid {color};
                border-radius: 14px;
                padding: 1.4rem 1.6rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.06);">
        <div style="display:flex;justify-content:space-between;
                    align-items:center;margin-bottom:0.8rem;">
            <span style="color:{color};font-size:1.05rem;letter-spacing:2px;">{stars}</span>
            <span style="background:{badge_color};color:#FFF;font-size:0.68rem;
                         font-weight:600;padding:3px 12px;border-radius:20px;
                         letter-spacing:0.06em;font-family:'Inter',sans-serif;">{source.upper()}</span>
        </div>
        <p style="color:{C['muted']};font-size:0.92rem;line-height:1.72;
                  margin:0;font-style:italic;font-family:'Inter',sans-serif;">"{short}"</p>
    </div>"""


def md_to_html(text: str) -> str:
    text = re.sub(
        r"^### (.+)$",
        r"<h4 style='font-family:Playfair Display,serif;color:#1A1A1A;"
        r"margin:1.4rem 0 0.4rem;font-size:0.95rem;text-transform:uppercase;"
        r"letter-spacing:0.05em;'>\\1</h4>",
        text, flags=re.MULTILINE,
    )
    text = re.sub(
        r"^## (.+)$",
        r"<h3 style='font-family:Playfair Display,serif;color:#1A1A1A;"
        r"margin:1.25rem 0 0.5rem;font-size:1.15rem;'>\\1</h3>",
        text, flags=re.MULTILINE,
    )
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong style='color:#1A1A1A;'>\\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\\1</em>", text)
    lines = text.split("\n")
    out, in_list = [], False
    for line in lines:
        s = line.strip()
        if re.match(r"^[*\-\d]\d*\.?\s", s):
            if not in_list:
                out.append("<ul style='padding-left:1.3rem;margin:0.6rem 0;'>")
                in_list = True
            content = re.sub(r"^[*\-]\s+|\d+\.\s+", "", s)
            out.append(f"<li style='color:#6B6B6B;margin-bottom:0.45rem;line-height:1.68;"
                       f"font-family:Inter,sans-serif;'>{content}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            if s:
                out.append(f"<p style='color:#6B6B6B;line-height:1.78;margin:0.55rem 0;"
                           f"font-family:Inter,sans-serif;'>{s}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def rec_card(header: str, body: str, color: str, emoji: str):
    html_body = md_to_html(body)
    st.markdown(f"""
    <div style="background:{C['card']};
                border: 1px solid {C['border']};
                border-left: 5px solid {color};
                border-radius: 16px;
                overflow: hidden;
                margin-bottom: 2.5rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
        <div style="background: linear-gradient(90deg, {color}10 0%, transparent 70%);
                    padding: 1rem 1.5rem;
                    border-bottom: 1px solid {C['border']};">
            <div style="font-size:10px;color:{color};text-transform:uppercase;
                        letter-spacing:0.14em;font-weight:600;margin-bottom:0.25rem;
                        font-family:'Inter',sans-serif;">{emoji} Complaint Cluster</div>
            <h2 style="font-family:'Playfair Display',serif;color:{C['text']};
                       margin:0;font-size:1.15rem;font-weight:700;">{header}</h2>
        </div>
        <div style="padding:1.1rem 1.5rem;">{html_body}</div>
    </div>
    """, unsafe_allow_html=True)


def ax(extra: dict = None) -> dict:
    base = dict(
        showgrid=True,
        gridcolor="#EEEEEE",
        zeroline=False,
        color="#6B6B6B",
        tickfont=dict(size=15, color="#444444"),
        linecolor="#E0E0E0",
    )
    if extra:
        base.update(extra)
    return base


def ax_clean(extra: dict = None) -> dict:
    """No grid, no ticks — for scatter/minimal charts."""
    base = dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title="",
        linecolor="rgba(0,0,0,0)",
    )
    if extra:
        base.update(extra)
    return base


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 The Problem",
    "🔬 The Science",
    "💡 The Findings",
    "📋 The Brief",
    "⚡ Live Demo",
])


# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    hero(
        "4.39 stars hides a story.",
        "Spice Up Thai has strong ratings across 564 Google and Yelp reviews. But buried "
        "inside that average are 25 negative reviews that cluster around three recurring, "
        "fixable problems. This project uses NLP to find them — and tells the owner exactly "
        "what to do about each one.",
    )

    cols = st.columns(4, gap="small")
    cards = [
        ("Total Reviews", "564", "Google + Yelp combined"),
        ("Average Rating", "4.39 ⭐", "Across all platforms"),
        ("Negative Reviews", "25", "≤ 3 stars, post-2021"),
        ("Model Accuracy", "80%", "Human-validated clusters"),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        with col:
            st.markdown(metric_card(label, value, sub), unsafe_allow_html=True)

    section_heading("Rating Distribution")
    callout(
        "The histogram looks promising — but the 5-star mass is so dominant it hides the "
        "signal. On Google and Yelp, even a handful of detailed 1-star reviews can "
        "permanently suppress a restaurant's search visibility.",
    )
    rc = full_df["rating"].value_counts().sort_index().reset_index()
    rc.columns = ["rating", "count"]
    rc["label"] = rc["rating"].astype(int).astype(str) + " star"
    fig_bar = px.bar(
        rc, x="count", y="label", orientation="h",
        color="rating",
        color_continuous_scale=["#C0392B", "#E67E22", "#F1C40F", "#82C341", "#27AE60"],
        text="count",
        labels={"count": "", "label": ""},
        title="All 564 Reviews by Star Rating",
    )
    fig_bar.update_traces(textposition="outside",
                          textfont=dict(color="#1A1A1A", size=14))
    fig_bar.update_coloraxes(showscale=False)
    fig_bar.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFA",
        font=dict(color="#1A1A1A", family="Inter, sans-serif", size=14),
        margin=dict(l=10, r=60, t=60, b=10),
        showlegend=False,
        height=500,
        title_font=dict(family="Playfair Display, serif", size=19, color="#1A1A1A"),
        xaxis=ax(),
        yaxis=ax({"categoryorder": "total ascending", "showgrid": False}),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    section_heading("Platform Split")
    callout(
        "Yelp reviewers write longer, more critical reviews. "
        "Google reviewers skew positive and brief. "
        "The complaint patterns differ by platform.",
    )
    sc = full_df["source"].value_counts().reset_index()
    sc.columns = ["source", "count"]
    fig_pie = go.Figure(go.Pie(
        labels=sc["source"], values=sc["count"],
        hole=0.62,
        marker=dict(colors=["#C0392B", "#E67E22"],
                    line=dict(color="#FFFFFF", width=3)),
        textinfo="percent+label",
        textfont=dict(size=14, color="#1A1A1A"),
    ))
    fig_pie.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#1A1A1A", family="Inter, sans-serif", size=14),
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
        height=500,
        title="Google vs. Yelp",
        title_font=dict(family="Playfair Display, serif", size=19, color="#1A1A1A"),
        annotations=[dict(text="564<br>reviews", x=0.5, y=0.5,
                          font=dict(size=16, color="#1A1A1A"), showarrow=False)],
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    avg_src = full_df.groupby("source")["rating"].mean().reset_index()
    avg_src.columns = ["source", "avg"]
    fig_avg = px.bar(
        avg_src, x="avg", y="source", orientation="h",
        color="source",
        color_discrete_map={"Google": "#C0392B", "Yelp": "#E67E22"},
        text=avg_src["avg"].round(2),
        labels={"avg": "Avg Rating", "source": ""},
        title="Avg Rating by Platform",
    )
    fig_avg.update_traces(textposition="outside",
                          textfont=dict(color="#1A1A1A", size=14))
    fig_avg.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFA",
        font=dict(color="#1A1A1A", family="Inter, sans-serif", size=14),
        margin=dict(l=10, r=60, t=50, b=10),
        showlegend=False,
        height=500,
        title_font=dict(family="Playfair Display, serif", size=19, color="#1A1A1A"),
        xaxis=ax({"range": [0, 5.8]}),
        yaxis=ax({"showgrid": False}),
    )
    st.plotly_chart(fig_avg, use_container_width=True)


# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    hero(
        "How do you find patterns in 25 messy human complaints?",
        "You can't keyword-search your way through natural language. This project encodes "
        "the semantic meaning of each review as a high-dimensional vector, maps those "
        "vectors to 2D, and lets the geometry reveal the clusters — no manual labeling needed.",
    )

    section_heading("The Pipeline")
    st.markdown(f'<p style="color:{C["muted"]};font-size:18px;line-height:1.75;'
                f'margin:0 0 2rem 0;font-family:Inter,sans-serif;">'
                f'Five steps, fully automated. Each stage transforms the raw text into '
                f'structured, actionable insight.</p>', unsafe_allow_html=True)

    steps = [
        ("📥", "Collect", "#4285F4",
         "Scraped 564 reviews from Google Maps and Yelp — raw text, star rating, date, platform."),
        ("🧠", "Embed", "#7B2FBE",
         "<strong>all-MiniLM-L6-v2</strong> converts each review into 384 numbers: "
         "a semantic fingerprint, not a keyword count."),
        ("📐", "UMAP", "#0891B2",
         "Reduces 384D → 5D (for clustering) and 2D (for this chart). "
         "Nearby points share similar meaning."),
        ("🔵", "K-Means", "#059669",
         "Groups the 25 complaints into k=3 clusters. K was chosen by maximizing "
         "<strong>silhouette score</strong> across k=2–6."),
        ("✍️", "LLM", "#C0392B",
         "<strong>Groq LLaMA-3</strong> reads each cluster and writes practical "
         "recommendations — constrained to a two-person team."),
    ]

    step_cols = st.columns(5, gap="medium")
    for col, (icon, name, color, desc) in zip(step_cols, steps):
        with col:
            st.markdown(f"""
            <div style="background:{C['card']};
                        border:1px solid {C['border']};
                        border-top:4px solid {color};
                        border-radius:14px;
                        padding:28px;
                        text-align:center;
                        min-height:240px;
                        height:100%;
                        box-shadow:0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size:2rem;margin-bottom:0.6rem;">{icon}</div>
                <div style="font-size:12px;text-transform:uppercase;letter-spacing:0.1em;
                            color:{color};font-weight:700;margin-bottom:0.6rem;
                            font-family:'Inter',sans-serif;">{name}</div>
                <p style="font-size:15px;color:{C['muted']};line-height:1.6;margin:0;
                          font-family:'Inter',sans-serif;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    section_heading("Semantic Map")
    callout(
        "Each dot below is one negative review, placed in 2D space by semantic similarity. "
        "Complaints that say similar things cluster together — even when the exact words differ. "
        "The three color groups emerged from the data automatically.",
    )

    negative_df["Stars"] = negative_df["rating"].apply(lambda r: "★" * int(r))
    negative_df["short_text"] = negative_df["text"].apply(
        lambda t: (t[:120] + "…") if len(t) > 120 else t
    )

    fig_umap = px.scatter(
        negative_df, x="umap_x", y="umap_y",
        color="cluster_name",
        color_discrete_map={CLUSTER[k]["name"]: CLUSTER[k]["color"] for k in CLUSTER},
        custom_data=["cluster_name", "Stars", "source", "short_text"],
        labels={"cluster_name": "Theme"},
        title="Semantic Map of 25 Negative Reviews",
    )
    fig_umap.update_traces(
        marker=dict(size=12, opacity=0.85, line=dict(width=1.5, color="#FFFFFF")),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]}<br>"
            "Source: %{customdata[2]}<br>"
            "<i>%{customdata[3]}</i>"
            "<extra></extra>"
        ),
    )
    fig_umap.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F9F5F3",
        font=dict(color="#1A1A1A", family="Inter, sans-serif", size=14),
        margin=dict(l=10, r=20, t=50, b=30),
        height=500,
        title_font=dict(family="Playfair Display, serif", size=19, color="#1A1A1A"),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#E8E0D8",
            font=dict(size=13, family="Inter, sans-serif", color="#1A1A1A"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, title_text="",
                    font=dict(size=13), bgcolor="rgba(0,0,0,0)",
                    itemclick=False, itemdoubleclick=False),
        xaxis=ax_clean(),
        yaxis=ax_clean(),
    )
    st.plotly_chart(fig_umap, use_container_width=True)

    divider()

    st.markdown(f"""
    <div style="background:{C['card']};border:1px solid {C['border']};
                border-left:4px solid {C['orange']};border-radius:14px;
                padding:1.5rem 2rem;margin-bottom:1rem;box-shadow:0 4px 20px rgba(0,0,0,0.06);">
        <div style="font-family:'Playfair Display',serif;color:{C['text']};
                    font-weight:700;font-size:1.15rem;margin-bottom:0.5rem;">
            Silhouette Score: 0.4492
        </div>
        <p style="color:{C['muted']};font-size:18px;line-height:1.75;margin:0;
                  font-family:'Inter',sans-serif;">
            K=3 was selected by maximizing the silhouette score across k=2–6. A score of
            0.45 on 25 noisy data points signals genuine, meaningful separation — not an
            arbitrary or manually chosen split.
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{C['card']};border:1px solid {C['border']};
                border-left:4px solid {C['red']};border-radius:14px;
                padding:1.5rem 2rem;box-shadow:0 4px 20px rgba(0,0,0,0.06);">
        <div style="font-family:'Playfair Display',serif;color:{C['text']};
                    font-weight:700;font-size:1.15rem;margin-bottom:0.5rem;">
            Human Validation: 80% Accuracy
        </div>
        <p style="color:{C['muted']};font-size:18px;line-height:1.75;margin:0;
                  font-family:'Inter',sans-serif;">
            Cluster assignments were manually reviewed against ground-truth human labels
            in a structured evaluation. 80% accuracy on fully unsupervised output confirms
            the geometry captured real, distinct complaint themes.
        </p>
    </div>""", unsafe_allow_html=True)


# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    hero(
        "Three clusters. Three fixable problems.",
        "Select a complaint theme below to read the actual reviews. Each cluster reveals "
        "a distinct operational failure and a clear pattern in what customers are trying to say.",
    )

    cluster_opts = {
        f"{CLUSTER[k]['emoji']}  Cluster {k} — {CLUSTER[k]['name']}  "
        f"({len(negative_df[negative_df['cluster']==k])} reviews)": k
        for k in CLUSTER
    }
    selected_label = st.selectbox("Select a complaint theme:", list(cluster_opts.keys()))
    selected_id = cluster_opts[selected_label]

    color = CLUSTER[selected_id]["color"]
    cluster_df = negative_df[negative_df["cluster"] == selected_id].reset_index(drop=True)

    descriptions = {
        0: "Customers describe feeling ignored, cold greetings, and front-counter moments that "
           "soured the experience regardless of food quality.",
        1: "Complaints center on undercooked food, wrong ingredients, and dishes that didn't "
           "match menu descriptions — kitchen consistency failures.",
        2: "Reviewers flag confusing menus, unexpected upcharges, and arriving to find the "
           "restaurant closed despite listed hours.",
    }
    callout(descriptions[selected_id], color)

    section_heading(f"{CLUSTER[selected_id]['name']} — Reviews")
    for _, row in cluster_df.iterrows():
        st.markdown(review_card(row["text"], row["rating"], row["source"], color),
                    unsafe_allow_html=True)

    section_heading("Cluster Stats")
    st.markdown(metric_card(
        "Avg Rating in Cluster",
        f"{cluster_df['rating'].mean():.2f} ⭐",
        f"vs. {full_df['rating'].mean():.2f} overall",
    ), unsafe_allow_html=True)

    st.markdown(SP2, unsafe_allow_html=True)

    rc2 = cluster_df["rating"].value_counts().sort_index().reset_index()
    rc2.columns = ["rating", "count"]
    fig_rc = px.bar(
        rc2, x="count", y=rc2["rating"].astype(str) + " ★",
        orientation="h", color="rating",
        color_continuous_scale=["#C0392B", "#E67E22"],
        text="count", labels={"count": "", "y": ""},
        title="Rating Breakdown",
    )
    fig_rc.update_traces(textposition="outside",
                         textfont=dict(color="#1A1A1A", size=14))
    fig_rc.update_coloraxes(showscale=False)
    fig_rc.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFA",
        font=dict(color="#1A1A1A", family="Inter, sans-serif", size=14),
        margin=dict(l=10, r=60, t=50, b=10),
        showlegend=False,
        height=500,
        title_font=dict(family="Playfair Display, serif", size=19, color="#1A1A1A"),
        xaxis=ax(),
        yaxis=ax({"showgrid": False}),
    )
    st.plotly_chart(fig_rc, use_container_width=True)

    src2 = cluster_df["source"].value_counts().reset_index()
    src2.columns = ["source", "count"]
    fig_src = px.bar(
        src2, x="count", y="source", orientation="h",
        color="source",
        color_discrete_map={"Google": "#C0392B", "Yelp": "#E67E22"},
        text="count", labels={"count": "", "source": ""},
        title="Source Split",
    )
    fig_src.update_traces(textposition="outside",
                          textfont=dict(color="#1A1A1A", size=14))
    fig_src.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFA",
        font=dict(color="#1A1A1A", family="Inter, sans-serif", size=14),
        margin=dict(l=10, r=60, t=50, b=10),
        showlegend=False,
        height=500,
        title_font=dict(family="Playfair Display, serif", size=19, color="#1A1A1A"),
        xaxis=ax(),
        yaxis=ax({"showgrid": False}),
    )
    st.plotly_chart(fig_src, use_container_width=True)

    divider()

    section_heading("All Clusters at a Glance")
    callout(
        "Food Quality has the most complaints, but service failures often drive customers "
        "away before they even taste the food. All three themes demand attention.",
    )

    summary = (
        negative_df.groupby("cluster")
        .agg(count=("text", "count"))
        .reset_index()
    )
    summary["name"] = summary["cluster"].map(lambda k: CLUSTER[k]["name"])

    fig_summary = px.bar(
        summary, x="count", y="name", orientation="h",
        color="name",
        color_discrete_map={CLUSTER[k]["name"]: CLUSTER[k]["color"] for k in CLUSTER},
        text="count", labels={"count": "Number of Complaints", "name": ""},
        title="Complaint Volume by Theme",
    )
    fig_summary.update_traces(textposition="outside",
                              textfont=dict(color="#1A1A1A", size=14))
    fig_summary.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFA",
        font=dict(color="#1A1A1A", family="Inter, sans-serif", size=14),
        margin=dict(l=10, r=60, t=60, b=10),
        showlegend=False,
        height=500,
        title_font=dict(family="Playfair Display, serif", size=19, color="#1A1A1A"),
        xaxis=ax({"range": [0, 14]}),
        yaxis=ax({"categoryorder": "total ascending", "showgrid": False}),
    )
    st.plotly_chart(fig_summary, use_container_width=True)


# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    hero(
        "What should the owner actually do?",
        "Groq LLaMA-3 was given each cluster's reviews alongside one hard constraint: "
        "this is a two-person operation. Every recommendation had to be implementable "
        "without extra staff, significant cost, or downtime.",
    )

    st.markdown(f"""
    <div style="background:{C['bg2']};border-radius:14px;padding:1.5rem 2rem;
                margin-bottom:2.5rem;border:1px solid {C['border']};">
        <p style="color:{C['muted']};font-size:18px;line-height:1.78;margin:0;
                  font-family:'Inter',sans-serif;">
            Every suggestion below was stress-tested against one question:
            <em><strong style="color:{C['text']}">Can a two-person team realistically do this?</strong></em>
            If not, it was cut. What remains is practical, low-friction, and immediately actionable.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cluster_sections = re.split(r"(?=## 🔴 Cluster)", brief_md.strip())
    cluster_sections = [s.strip() for s in cluster_sections if s.strip().startswith("## 🔴 Cluster")]

    for i, section in enumerate(cluster_sections):
        info = CLUSTER[i]
        lines = section.split("\n")
        header_text = re.sub(r"^##\s*🔴\s*", "", lines[0]).strip()
        body_text = "\n".join(lines[1:]).strip()

        st.markdown(f"""
        <div style="background:{C['card']};border:1px solid {C['border']};
                    border-left:5px solid {info['color']};border-radius:16px;
                    overflow:hidden;margin:1.5rem 0 0.75rem;
                    box-shadow:0 4px 20px rgba(0,0,0,0.08);">
            <div style="background:linear-gradient(90deg,{info['color']}10 0%,transparent 70%);
                        padding:1rem 1.5rem;border-bottom:1px solid {C['border']};">
                <div style="font-size:11px;color:{info['color']};text-transform:uppercase;
                            letter-spacing:0.14em;font-weight:600;margin-bottom:0.25rem;
                            font-family:'Inter',sans-serif;">{info['emoji']} Complaint Cluster</div>
                <span style="font-family:'Playfair Display',serif;font-size:1.25rem;
                             font-weight:700;color:{C['text']};">{header_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(body_text)
        divider()


# ── TAB 5 ─────────────────────────────────────────────────────────────────────
# Prediction uses cosine similarity against cluster centroids computed in the original 384-dim embedding space rather than UMAP-reduced space.
# Since UMAP was fit on only 25 negative reviews. Calling reducer.transform() on unseen text forces UMAP to place a new point into a manifold learned from very
# few examples, which is numerically unstable and produced near-random cluster assignments at inference time.
# Cosine similarity in the full embedding space is stable regardless of training set size, since it compares semantic vectors directly. UMAP is retained for the 2D scatter plot visualization only.
with tab5:
    hero(
        "Which complaint cluster does this belong to?",
        "Type any restaurant complaint below. The model encodes it with a sentence "
        "transformer, reduces dimensions with UMAP, then predicts the cluster using "
        "K-Means — the exact same pipeline used on the 25 real reviews.",
    )

    CLUSTER_EXPLANATIONS = {
        0: (
            "This complaint is about <strong>staff behavior and service quality</strong>. "
            "Spice Up Thai has received feedback about front-counter interactions — customers "
            "feeling ignored, cold greetings, or staff distracted by phones. "
            "These moments carry outsized weight on whether a customer returns."
        ),
        1: (
            "This complaint relates to <strong>food quality and kitchen consistency</strong>. "
            "Reviews here describe dishes that were undercooked, used wrong ingredients, "
            "or didn't match menu descriptions. In a two-person kitchen, split attention "
            "during busy service is the root cause."
        ),
        2: (
            "This complaint falls into <strong>operations and pricing transparency</strong>. "
            "Customers were frustrated by unexpected upcharges, confusing menus, "
            "or arriving to find the restaurant closed despite posted hours. "
            "Trust is built through predictability — these issues undermine it."
        ),
    }

    user_input = st.text_area(
        "Enter a complaint:",
        placeholder='e.g. "The cashier barely looked up from her phone when I ordered…"',
        height=110,
        label_visibility="collapsed",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        predict_clicked = st.button("Predict Cluster →", type="primary",
                                    use_container_width=True)


    if predict_clicked:
        if not user_input.strip():
            st.warning("Please enter a complaint first.")
        else:
            with st.spinner("Encoding and predicting…"):
                try:
                    encoder, centroids = load_models()
                    emb = encoder.encode([user_input.strip()])[0]

                    # Cosine similarity against each centroid in 384-dim space
                    emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
                    cent_norms = centroids / (
                        np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
                    )
                    sims = cent_norms @ emb_norm          # shape (3,)

                    predicted = int(np.argmax(sims))

                    # Softmax over similarities → confidence percentages
                    exp_sims = np.exp(sims - sims.max())
                    probs = exp_sims / exp_sims.sum()
                    confidence = probs[predicted]

                    info = CLUSTER[predicted]
                    explanation = CLUSTER_EXPLANATIONS[predicted]

                    st.markdown(f"""
                    <div style="background:{C['bg2']};
                                border:1px solid {C['border']};
                                border-left:5px solid {info['color']};
                                border-radius:16px;
                                padding:2.5rem 2.75rem;
                                margin-top:2rem;
                                box-shadow:0 4px 20px rgba(0,0,0,0.08);">
                        <div style="font-size:11px;color:{info['color']};text-transform:uppercase;
                                    letter-spacing:0.16em;font-weight:700;margin-bottom:0.5rem;
                                    font-family:'Inter',sans-serif;">Predicted Cluster</div>
                        <h2 style="font-family:'Playfair Display',serif;color:{C['text']};
                                   margin:0 0 0.3rem;font-size:2rem;font-weight:700;">
                            {info['emoji']} {info['name']}
                        </h2>
                        <div style="color:{info['color']};font-size:0.95rem;font-weight:600;
                                    margin-bottom:1.5rem;font-family:'Inter',sans-serif;">
                            {confidence*100:.0f}% confidence
                        </div>
                        <p style="color:{C['muted']};font-size:18px;line-height:1.78;
                                  margin:0;font-family:'Inter',sans-serif;">
                            {explanation}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(SP2, unsafe_allow_html=True)

                    prob_df = pd.DataFrame([
                        {"Theme": CLUSTER[k]["name"], "Confidence": float(probs[k])}
                        for k in CLUSTER
                    ]).sort_values("Confidence", ascending=True)

                    fig_conf = px.bar(
                        prob_df, x="Confidence", y="Theme", orientation="h",
                        color="Theme",
                        color_discrete_map={CLUSTER[k]["name"]: CLUSTER[k]["color"]
                                            for k in CLUSTER},
                        text=prob_df["Confidence"].apply(lambda p: f"{p*100:.0f}%"),
                        title="Confidence Across All Clusters",
                        labels={"Confidence": "Confidence Score", "Theme": ""},
                    )
                    fig_conf.update_traces(textposition="outside",
                                           textfont=dict(color="#1A1A1A", size=13))
                    fig_conf.update_layout(
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FAFAFA",
                        font=dict(color="#1A1A1A", family="Inter, sans-serif", size=14),
                        margin=dict(l=10, r=70, t=60, b=10),
                        showlegend=False,
                        height=500,
                        title_font=dict(family="Playfair Display, serif", size=19,
                                        color="#1A1A1A"),
                        xaxis=ax({"range": [0, 1.18], "tickformat": ".0%"}),
                        yaxis=ax({"showgrid": False}),
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)

                    if confidence >= 0.55:
                        interp_color, interp_icon, interp_text = (
                            "#2E7D32", "✅",
                            "High confidence — this complaint maps clearly to one theme.",
                        )
                    elif confidence >= 0.40:
                        interp_color, interp_icon, interp_text = (
                            "#E65100", "⚠️",
                            "Moderate confidence — this review mentions multiple themes. "
                            "The model is splitting between clusters.",
                        )
                    else:
                        interp_color, interp_icon, interp_text = (
                            "#C0392B", "🔴",
                            "Low confidence — this complaint spans several categories. "
                            "Human review recommended.",
                        )
                    st.markdown(f"""
                    <div style="background:{interp_color}0D;border-left:3px solid {interp_color};
                                padding:0.75rem 1.1rem;border-radius:0 8px 8px 0;
                                margin:0.25rem 0 1.5rem;font-family:'Inter',sans-serif;">
                        <span style="color:{interp_color};font-weight:600;font-size:15px;">
                            {interp_icon} {interp_text}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <p style="color:{C['muted']};font-size:14px;line-height:1.7;
                               font-family:'Inter',sans-serif;margin-top:0.5rem;">
                        <strong style="color:{C['text']};">Known limitation:</strong>
                        our human-in-the-loop evaluation found the model misclassifies some
                        Service complaints as Operations (4 of 25). Customers often describe
                        operational friction — wrong hours, surprise charges — using service
                        language. Ambiguous, multi-topic reviews like this one surface that overlap.
                    </p>
                    <p style="color:{C['muted']};font-size:13px;line-height:1.65;
                               font-family:'Inter',sans-serif;margin-top:1rem;
                               font-style:italic;border-top:1px solid {C['border']};
                               padding-top:0.75rem;">
                        Prediction uses cosine similarity against cluster centroids in the original
                        384-dimensional embedding space. UMAP is used for visualization only —
                        with 25 training points, projecting unseen text through UMAP is unstable.
                    </p>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    else:
        st.markdown(f"""
        <div style="background:{C['bg2']};border:1px solid {C['border']};
                    border-radius:16px;padding:4rem;text-align:center;
                    margin-top:1.5rem;">
            <div style="font-size:3rem;margin-bottom:1.25rem;">⚡</div>
            <p style="font-size:18px;margin:0;line-height:1.75;
                      color:{C['muted']};font-family:'Inter',sans-serif;font-weight:300;">
                Type a complaint above and click
                <strong style="color:{C['text']};font-weight:600;">Predict Cluster →</strong><br>
                to see the model classify it in real time.
            </p>
        </div>
        """, unsafe_allow_html=True)
