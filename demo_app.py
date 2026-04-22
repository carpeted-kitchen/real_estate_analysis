import warnings
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pathlib

"""
Retirement Home Recommender — Streamlit App
Reads Suitability_score_house.csv (output of re_retirement_rank.py)
and surfaces the top 5 properties for a user's chosen state & budget.
"""

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetireRight · Home Finder",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0d1117;
    color: #e8e3dc;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #131920 !important;
    border-right: 1px solid #1e2d3d;
}
section[data-testid="stSidebar"] * {
    color: #c8bfb0 !important;
}

/* Headings */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #f0ead8 !important;
}

/* Hero title */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    color: #f0ead8;
    line-height: 1.1;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    color: #7a9e7e;
    font-weight: 300;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.metric-card {
    background: #131920;
    border: 1px solid #1e2d3d;
    border-radius: 10px;
    padding: 16px 22px;
    flex: 1;
    min-width: 140px;
}
.metric-card .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a7c59;
    margin-bottom: 4px;
}
.metric-card .value {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f0ead8;
}

/* Property cards */
.prop-card {
    background: linear-gradient(135deg, #131920 0%, #0f1a14 100%);
    border: 1px solid #1e2d3d;
    border-left: 4px solid #4a7c59;
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 14px;
    position: relative;
    transition: border-color 0.2s;
}
.prop-card:hover {
    border-left-color: #7a9e7e;
}
.prop-rank {
    position: absolute;
    top: 18px;
    right: 22px;
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 900;
    color: #1e2d3d;
    line-height: 1;
}
.prop-street {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #f0ead8;
    margin-bottom: 4px;
}
.prop-location {
    font-size: 0.85rem;
    color: #7a9e7e;
    margin-bottom: 14px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.prop-stats {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
}
.prop-stat {
    display: flex;
    flex-direction: column;
}
.prop-stat .stat-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4a6650;
    margin-bottom: 2px;
}
.prop-stat .stat-value {
    font-size: 0.95rem;
    font-weight: 500;
    color: #c8bfb0;
}
.score-badge {
    display: inline-block;
    background: #4a7c59;
    color: #f0ead8;
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 20px;
    margin-top: 10px;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1e2d3d;
    margin: 2rem 0;
}

/* Info box */
.info-box {
    background: #0f1a14;
    border: 1px solid #1e2d3d;
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 0.85rem;
    color: #7a9e7e;
    margin-bottom: 1.5rem;
}

/* Streamlit selectbox / slider labels */
label {
    color: #c8bfb0 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em !important;
}

/* Buttons */
.stButton > button {
    background: #4a7c59 !important;
    color: #f0ead8 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.6rem !important;
    letter-spacing: 0.03em !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #5c9470 !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #131920;
    border-radius: 8px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #7a9e7e !important;
    background: transparent !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}
.stTabs [aria-selected="true"] {
    background: #4a7c59 !important;
    color: #f0ead8 !important;
}

/* Progress bars */
.stProgress > div > div {
    background: #4a7c59 !important;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #1e2d3d !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    drop_cols = [c for c in ["status", "city", "brokered_by", "zip_code", "prev_sold_date"] if c in df.columns]
    X = df.drop(drop_cols + ["Suitability"], axis=1, errors="ignore")
    Y = df["Suitability"]
    X = X.select_dtypes(include=[np.number])

    train_x, test_x, train_y, test_y = train_test_split(
        X, Y, test_size=0.3, random_state=42, shuffle=True
    )

    # Linear Regression
    lr = LinearRegression()
    lr.fit(train_x, train_y)
    lr_r2 = lr.score(test_x, test_y)

    # KNN
    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(train_x)

    return lr, knn, train_x, train_y, lr_r2


def knn_predict(knn, train_x, train_y, query_rows):
    preds = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, row in query_rows.iterrows():
            dists, idxs = knn.kneighbors([row], return_distance=True)
            neighbor_scores = train_y.iloc[idxs[0]].values
            # Inverse-distance weighting
            weights = 1 / (dists[0] + 1e-9)
            preds.append(np.average(neighbor_scores, weights=weights))
    return np.array(preds)


def get_recommendations(df, lr_model, knn_model, train_x, train_y,
                        state_filter, budget, model_choice, top_n=5):
    # Reconstruct state column from OHE if needed
    state_cols = [c for c in df.columns if c.startswith("state_")]
    if state_cols:
        # Reverse OHE to find state
        ohe_part = df[state_cols]
        df = df.copy()
        df["_state"] = ohe_part.idxmax(axis=1).str.replace("state_", "")
    else:
        df = df.copy()
        df["_state"] = df.get("state", "Unknown")

    subset = df[(df["_state"] == state_filter) & (df["price"] <= budget)].copy()

    if subset.empty:
        return None, 0

    drop_cols = [c for c in ["status", "city", "brokered_by", "zip_code",
                             "prev_sold_date", "Suitability", "_state"] if c in subset.columns]
    X_sub = subset.drop(drop_cols, axis=1, errors="ignore").select_dtypes(include=[np.number])
    X_sub = X_sub.reindex(columns=train_x.columns, fill_value=0)

    if model_choice == "Linear Regression":
        subset["pred_score"] = lr_model.predict(X_sub)
    else:
        subset["pred_score"] = knn_predict(knn_model, train_x, train_y, X_sub)

    top = subset.nlargest(top_n, "pred_score").copy()
    return top, len(subset)


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏡 RetireRight")
    st.markdown("<hr style='border-color:#1e2d3d;margin:8px 0 20px'>", unsafe_allow_html=True)

    # File picker
    csv_path = st.text_input(
        "Scored CSV path",
        value="Suitability_score_house.csv",
        help="Path to the output of re_retirement_rank.py"
    )

    st.markdown("---")
    st.markdown("**Search Preferences**")

    model_choice = st.radio(
        "Scoring model",
        ["Linear Regression", "KNN (weighted)"],
        help="Linear Regression is faster; KNN captures non-linear patterns."
    )

    top_n = st.slider("Results to show", min_value=3, max_value=10, value=5)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#4a6650;line-height:1.6'>"
        "Data powered by VT demographic & environment datasets. "
        "Scores reflect price, size, crime, weather, air quality, "
        "walkability, & healthcare."
        "</div>",
        unsafe_allow_html=True
    )

# ── Main layout ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Find Your Perfect<br>Retirement Home</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered suitability ranking · 11-factor scoring</div>', unsafe_allow_html=True)

# Load data
if not pathlib.Path(csv_path).exists():
    st.error(f"❌ Cannot find `{csv_path}`. Run `re_retirement_rank.py` first to generate the scored dataset.")
    st.stop()

with st.spinner("Loading and training models…"):
    df = load_data(csv_path)
    lr_model, knn_model, train_x, train_y, lr_r2 = train_models(df)

# Recover state list from OHE columns
state_cols = [c for c in df.columns if c.startswith("state_")]
if state_cols:
    available_states = sorted([c.replace("state_", "") for c in state_cols])
else:
    available_states = sorted(df["state"].dropna().unique().tolist()) if "state" in df.columns else ["Unknown"]

# ── Filter controls ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    selected_state = st.selectbox(
        "📍 Preferred state",
        available_states,
        index=available_states.index("New York") if "New York" in available_states else 0
    )

with col2:
    budget = st.slider(
        "💰 Maximum budget",
        min_value=int(df["price"].min()),
        max_value=min(int(df["price"].max()), 2_000_000),
        value=450_000,
        step=10_000,
        format="$%d"
    )

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    search_clicked = st.button("Search →", use_container_width=True)

# ── Run recommendation ──────────────────────────────────────────────────────────
if search_clicked or True:  # auto-run on load/change
    with st.spinner("Finding best matches…"):
        top_homes, pool_size = get_recommendations(
            df, lr_model, knn_model, train_x, train_y,
            selected_state, budget, model_choice, top_n
        )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if top_homes is None or top_homes.empty:
        st.warning(
            f"No properties found in **{selected_state}** under **${budget:,}**. Try raising your budget or selecting a different state.")
    else:
        # ── Summary metrics ────────────────────────────────────────────────────
        avg_suit = top_homes["pred_score"].mean()
        avg_price = top_homes["price"].mean()
        avg_crime = top_homes["crime"].mean() if "crime" in top_homes.columns else None
        avg_temp = top_homes["avg_temp"].mean() if "avg_temp" in top_homes.columns else None

        st.markdown(
            f'<div class="info-box">'
            f'Found <strong>{pool_size:,}</strong> properties in <strong>{selected_state}</strong> '
            f'under <strong>${budget:,}</strong> — showing top {min(top_n, len(top_homes))} by {model_choice} score.'
            f'</div>',
            unsafe_allow_html=True
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="metric-card"><div class="label">Avg Suitability</div>'
                f'<div class="value">{avg_suit:.3f}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(
                f'<div class="metric-card"><div class="label">Avg Price</div>'
                f'<div class="value">${avg_price:,.0f}</div></div>', unsafe_allow_html=True)
        with m3:
            val = f"{avg_crime:.0f}" if avg_crime is not None else "N/A"
            st.markdown(
                f'<div class="metric-card"><div class="label">Avg Crime Risk</div>'
                f'<div class="value">{val}</div></div>', unsafe_allow_html=True)
        with m4:
            val = f"{avg_temp:.1f}°F" if avg_temp is not None else "N/A"
            st.markdown(
                f'<div class="metric-card"><div class="label">Avg Temp</div>'
                f'<div class="value">{val}</div></div>', unsafe_allow_html=True)

        # ── Tabs: Cards / Table / Details ──────────────────────────────────────
        tab1, tab2, tab3 = st.tabs(["🏠 Property Cards", "📊 Comparison Table", "📈 Score Breakdown"])

        with tab1:
            for rank, (_, row) in enumerate(top_homes.iterrows(), 1):
                street = str(row.get("street", "Address unavailable")).title()
                city = str(row.get("city", "—")).title() if "city" in row.index else "—"
                price = f"${row['price']:,.0f}"
                beds = int(row["bed"]) if "bed" in row.index and not pd.isna(row["bed"]) else "—"
                baths = row["bath"] if "bath" in row.index and not pd.isna(row["bath"]) else "—"
                size = f"{int(row['house_size']):,} sq ft" if "house_size" in row.index and not pd.isna(
                    row["house_size"]) else "—"
                score = row["pred_score"]
                crime_val = f"{row['crime']:.0f}" if "crime" in row.index else "—"
                temp_val = f"{row['avg_temp']:.1f}°F" if "avg_temp" in row.index else "—"
                walk_val = f"{row['walkability']:.1f}" if "walkability" in row.index else "—"

                st.markdown(f"""
                <div class="prop-card">
                    <div class="prop-rank">#{rank}</div>
                    <div class="prop-street">{street}</div>
                    <div class="prop-location">{city} · {selected_state}</div>
                    <div class="prop-stats">
                        <div class="prop-stat">
                            <span class="stat-label">Price</span>
                            <span class="stat-value">{price}</span>
                        </div>
                        <div class="prop-stat">
                            <span class="stat-label">Beds / Baths</span>
                            <span class="stat-value">{beds} bd · {baths} ba</span>
                        </div>
                        <div class="prop-stat">
                            <span class="stat-label">Size</span>
                            <span class="stat-value">{size}</span>
                        </div>
                        <div class="prop-stat">
                            <span class="stat-label">Crime Risk</span>
                            <span class="stat-value">{crime_val}</span>
                        </div>
                        <div class="prop-stat">
                            <span class="stat-label">Avg Temp</span>
                            <span class="stat-value">{temp_val}</span>
                        </div>
                        <div class="prop-stat">
                            <span class="stat-label">Walkability</span>
                            <span class="stat-value">{walk_val}</span>
                        </div>
                    </div>
                    <div class="score-badge">Suitability {score:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            display_cols = [c for c in [
                "street", "price", "bed", "bath", "house_size",
                "crime", "avg_temp", "walkability",
                "carbon_monoxide", "particulate_matter", "pred_score"
            ] if c in top_homes.columns]

            rename_map = {
                "street": "Address", "price": "Price ($)", "bed": "Beds",
                "bath": "Baths", "house_size": "Sq Ft", "crime": "Crime Risk",
                "avg_temp": "Avg Temp (°F)", "walkability": "Walkability",
                "carbon_monoxide": "CO", "particulate_matter": "PM10",
                "pred_score": "Suitability Score"
            }

            table_df = top_homes[display_cols].rename(columns=rename_map).reset_index(drop=True)
            table_df.index = table_df.index + 1

            fmt = {}
            if "Price ($)" in table_df.columns:
                fmt["Price ($)"] = "${:,.0f}"
            if "Suitability Score" in table_df.columns:
                fmt["Suitability Score"] = "{:.4f}"
            if "Sq Ft" in table_df.columns:
                fmt["Sq Ft"] = "{:,.0f}"

            st.dataframe(table_df.style.format(fmt), use_container_width=True)

        with tab3:
            st.markdown("#### Suitability score components")
            score_factors = {
                "crime": ("Crime Safety", True),
                "avg_temp": ("Temperature", False),
                "walkability": ("Walkability", False),
                "particulate_matter": ("Air Quality (PM10)", True),
                "carbon_monoxide": ("Air Quality (CO)", True),
                "Median Cash Rent": ("Median Rent", True),
                "Health Care": ("Healthcare Access", True),
                "weather_risk": ("Weather Risk", True),
            }

            for col, (label, lower_is_better) in score_factors.items():
                if col in top_homes.columns:
                    vals = top_homes[col].values
                    col1a, col1b, col1c = st.columns([2, 5, 1])
                    with col1a:
                        st.markdown(f"<span style='color:#7a9e7e;font-size:0.8rem'>{label}</span>",
                                    unsafe_allow_html=True)
                    with col1b:
                        col_min = df[col].min()
                        col_max = df[col].max()
                        avg_val = vals.mean()
                        if col_max > col_min:
                            norm = (avg_val - col_min) / (col_max - col_min)
                            if lower_is_better:
                                norm = 1 - norm
                        else:
                            norm = 0.5
                        st.progress(float(np.clip(norm, 0, 1)))
                    with col1c:
                        st.markdown(f"<span style='color:#c8bfb0;font-size:0.8rem'>{vals.mean():.1f}</span>",
                                    unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='info-box'>Model: <strong>{model_choice}</strong> · "
                f"LR R² on test set: <strong>{lr_r2:.4f}</strong> · "
                f"Training rows: <strong>{len(train_x):,}</strong></div>",
                unsafe_allow_html=True
            )
