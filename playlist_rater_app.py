import streamlit as st
import pandas as pd
import html


from playlist_backend import (
    sp,                    # authenticated Spotify client
    best_xgb_full,         # trained model
    best_threshold_full,   # F1-optimal threshold
    rate_playlist          # the function
)

st.set_page_config(page_title="Playlist Rater", page_icon="🎧", layout="wide")

# --- Custom CSS for styling ---
st.markdown("""
<style>

/* ---------- PAGE BACKGROUND & LAYOUT ---------- */

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #1f2937 0%, #020617 45%, #020617 100%);
}

.main .block-container {
    max-width: 900px;
    margin: 0 auto;
    padding-top: 2rem;
}

/* Kill the rounded "bubbles" between sections */
[data-testid="stVerticalBlock"] > div {
    background: transparent !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}

/* Remove default header background */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* ---------- TYPOGRAPHY ---------- */

h1, h2, h3 {
    font-family: 'Segoe UI', system-ui, sans-serif;
    color: #f9fafb !important;
}

p, label, span {
    color: #e5e7eb !important;
}

/* ---------- RATING CARD ---------- */

.card {
    background: #0f172a;
    border-radius: 18px;
    padding: 1.8rem 2rem;
    margin: 1.2rem 0;
    box-shadow: 0 12px 30px rgba(0,0,0,0.45);
    border: 1px solid rgba(148,163,184,0.35);
}

.big-rating {
    text-align: center;
    font-size: 4rem;
    font-weight: 800;
    margin-top: -10px;
    color: #facc15; /* gold */
}

.rating-label {
    text-align: center;
    font-size: 1.4rem;
    color: #f9fafb;
}
            
/* ---------- BUTTON ---------- */
.stButton > button {
    background-color: #0f172a !important;
    color: #f9fafb !important;
    border-radius: 999px;
    border: 1px solid #64748b;
    padding: 0.45rem 1.4rem;
    font-weight: 600;
    font-size: 0.95rem;
}

.stButton > button:hover {
    background-color: #1f2937 !important;
    border-color: #facc15 !important;
}

/* ---------- CUSTOM SONG TABLES ---------- */
.cool-table {
    width: 100%;
    border-collapse: collapse;
    background: #0f172a;
    color: #f1f5f9;
    border-radius: 12px;
    overflow: hidden;
    font-size: 0.95rem;
}

.cool-table thead {
    background: #111827;
}

.cool-table th, .cool-table td {
    padding: 10px 12px;
    text-align: left;
}

.cool-table th {
    font-weight: 600;
    border-bottom: 1px solid #1f2937;
}

.cool-table tr:nth-child(even) td {
    background: #020617;
}

.cool-table tr:nth-child(odd) td {
    background: #020818;
}

.cool-table tr:hover td {
    background: #1f2937;
}
        
/* Hide empty spacer block right under the button */
[data-testid="stVerticalBlock"] > div:empty {
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}

</style>
""", unsafe_allow_html=True)



st.title("🎧 Spotify Playlist Rater")
st.write(
    "Paste a public Spotify playlist URL below and see how basic it is"
    "based on the model from my machine learning project."
)

# --- Input ---

default_url = "https://open.spotify.com/playlist/5xbg1Y0wWC0daWLpq0cesL?si=02378546eecf4a02"

playlist_url = st.text_input(
    "Spotify playlist URL:",
    value=default_url,
    help="Make sure the playlist is public."
)

rate_button = st.button("Rate this playlist 🚀")

# --- When user clicks ---

if rate_button:
    if not playlist_url.strip():
        st.error("Please paste a valid Spotify playlist URL.")
    else:
        with st.spinner("Scoring your playlist..."):
            try:
                # 1) Model feature names
                model_features = list(best_xgb_full.get_booster().feature_names)

                # 2) Call your core function
                summary, top5, bottom5, df_scored = rate_playlist(
                    playlist_url=playlist_url,
                    sp=sp,
                    model=best_xgb_full,
                    model_features=model_features,
                    threshold=best_threshold_full,
                )

                # --- Big final rating section ---
                final_pct = summary.get("final_score_pct", 0.0)
                label = summary.get("label", "")

                st.markdown("<div class='card'>", unsafe_allow_html=True)

                st.markdown("<h2 style='text-align:center; color:#f9fafb;'>🎯 Playlist Rating</h2>", unsafe_allow_html=True)


                st.markdown(f"<div class='big-rating'>{final_pct:.1f}%</div>", unsafe_allow_html=True)

                st.markdown(f"<div class='rating-label'>{label}</div>", unsafe_allow_html=True)

                st.markdown(
                    f"<p style='text-align:center; color:#cbd5f5;'>Based on {len(df_scored)} tracks</p>",
                    unsafe_allow_html=True,
                )

                st.markdown("</div>", unsafe_allow_html=True)

                # --- Top 3 covers strip (from top5) ---
                top3 = top5.head(3)

                # Only show if we actually have image URLs
                if "album_image_url" in top3.columns:
                    st.markdown(
                        "<h3 style='text-align:center; color:#f9fafb; margin-top:1.5rem;'>"
                        "🔥 Top 3 Tracks (Cover Preview)"
                        "</h3>",
                        unsafe_allow_html=True,
                    )

                    cols = st.columns(3)

                    for i, (_, row) in enumerate(top3.iterrows()):
                        with cols[i]:
                            img_url = row["album_image_url"]
                            if img_url:
                                st.image(img_url, use_container_width=True)
                            st.markdown(
                                f"""
                                <div style='text-align:center; color:#f1f5f9; font-size:0.9rem; margin-top:0.5rem;'>
                                    <strong>{html.escape(str(row['track_name']))}</strong><br>
                                    <span style='color:#cbd5e1;'>{html.escape(str(row['artist_name']))}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )



                # --- Top 5 section ---
                def render_song_table(df, title_emoji, title_text):
                    rows = []
                    for _, row in df.iterrows():
                        track = html.escape(str(row["track_name"]))
                        artist = html.escape(str(row["artist_name"]))
                        year = html.escape(str(row["year"]))
                        score = f"{row['hit_score']:.3f}"

                        # no leading spaces → no Markdown code block
                        rows.append(
                            f"<tr>"
                            f"<td>{track}</td>"
                            f"<td>{artist}</td>"
                            f"<td>{year}</td>"
                            f"<td>{score}</td>"
                            f"</tr>"
                        )

                    table_html = (
                        "<div class='card'>"
                        f"<h3>{title_emoji} {title_text}</h3>"
                        "<table class='cool-table'>"
                        "<thead>"
                        "<tr>"
                        "<th>Track</th>"
                        "<th>Artist</th>"
                        "<th>Year</th>"
                        "<th>Hit score</th>"
                        "</tr>"
                        "</thead>"
                        "<tbody>"
                        + "".join(rows) +
                        "</tbody>"
                        "</table>"
                        "</div>"
                    )

                    st.markdown(table_html, unsafe_allow_html=True)



                # --- Top 5 section ---
                render_song_table(top5, "🔥", "Top 5 most 'hit-like' tracks")

                # --- Bottom 5 section ---
                render_song_table(bottom5, "🧊", "Bottom 5 least 'hit-like' tracks")


                # Optional: expandable full table
                with st.expander("See full scored playlist"):
                    st.dataframe(df_scored)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
