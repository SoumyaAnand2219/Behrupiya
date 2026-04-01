# app.py

import math
import time
import datetime
from io import BytesIO

import pandas as pd
import numpy as np
import streamlit as st
import mplfinance as mpf
import pyotp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tslearn.metrics import cdist_dtw
from SmartApi import SmartConnect

# 1️⃣ Your dict of base symbols → tokens
from nse_stock_tokens import stock_tokens  # e.g. {"FACT":"1008", ...}

st.set_page_config(page_title="Pattern Similarity Scanner", layout="wide")

# 2️⃣ Build the NSE-style ticker→token map
token_map = { f"{sym}.NS": tok for sym, tok in stock_tokens.items() }
all_symbols = list(token_map.keys())
total = len(all_symbols)

# 3️⃣ Batch selector (500 per batch)
batch_size = 500
n_batches  = math.ceil(total / batch_size)
batch_labels = [
    f"{i} ({(i-1)*batch_size+1}–{min(i*batch_size, total)})"
    for i in range(1, n_batches+1)
]
batch_choice = st.sidebar.selectbox("Batch #", batch_labels, index=0)
batch_num = int(batch_choice.split()[0])
symbols = all_symbols[(batch_num-1)*batch_size : batch_num*batch_size]

# 4️⃣ Cached Angel One login
@st.cache_resource
def angel_login():
    API_KEY     = "g5o6vfTl"
    CLIENT_ID   = "R59803990"
    PASSWORD    = "1234"
    TOTP_SECRET = "5W4MC6MMLANC3UYOAW2QDUIFEU"
    totp   = pyotp.TOTP(TOTP_SECRET).now()
    client = SmartConnect(api_key=API_KEY)
    client.generateSession(CLIENT_ID, PASSWORD, totp)
    return client

client = angel_login()

# 5️⃣ Cached OHLCV fetcher (0.3s throttle)
@st.cache_data
def fetch_price_data(token: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    params = {
        "exchange":    "NSE",
        "symboltoken": token,
        "interval":    "ONE_DAY",
        "fromdate":    f"{start_date:%Y-%m-%d} 00:00",
        "todate":      f"{end_date:%Y-%m-%d} 23:59"
    }
    resp = client.getCandleData(params)
    time.sleep(0.3)
    df = pd.DataFrame(resp["data"], columns=['Date','Open','High','Low','Close','Volume'])
    df['Date'] = (
        pd.to_datetime(df['Date'], unit='ms', errors='coerce')
          .fillna(pd.to_datetime(df['Date'], errors='coerce'))
    )
    df.set_index('Date', inplace=True)
    return df.dropna()

# 6️⃣ Feature extraction
def extract_features(df: pd.DataFrame) -> np.ndarray:
    op, cp = df['Open'].iloc[0], df['Close'].iloc[-1]
    hp, lp = df['High'].max(), df['Low'].min()
    vol    = df['Volume'].mean()
    pc     = (cp - op) / op
    vola   = (hp - lp) / op
    dr     = df['Close'].pct_change().fillna(0).values
    return np.hstack([pc, vola, (cp-op)/op, vol, dr])

# 7️⃣ Similarity: PCA + DTW blend
def compute_similarity(tf: np.ndarray, cf: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    cf_s   = scaler.fit_transform(cf)
    tf_s   = scaler.transform(tf.reshape(1, -1))
    pca    = PCA(n_components=min(10, cf_s.shape[1]))
    cf_p   = pca.fit_transform(cf_s)
    tf_p   = pca.transform(tf_s)
    pdist  = np.linalg.norm(cf_p - tf_p, axis=1)
    pn     = pdist / (pdist.max() or 1)
    tr     = tf[4:]
    cr     = cf[:,4:]
    dtw    = cdist_dtw(tr.reshape(1,-1), cr)[0]
    dn     = dtw / (dtw.max() or 1)
    return 1 - (0.5*pn + 0.5*dn)

# ── UI ─────────────────────────────────────────────────────────────────────
st.title("🔍 Pattern Similarity Scanner")
st.sidebar.markdown(f"**Total symbols:** {total} — Batch {batch_num}/{n_batches}")

with st.sidebar:
    st.header("Search & Dates")
    target_sym = st.selectbox("Target ticker", symbols)
    tgt_start  = st.date_input("Target start",     datetime.date.today()-datetime.timedelta(days=60))
    tgt_end    = st.date_input("Target end",       datetime.date.today())
    comp_start = st.date_input("Comparison start", datetime.date.today()-datetime.timedelta(days=365))
    comp_end   = st.date_input("Comparison end",   datetime.date.today())
    top_k      = st.slider("Top N unique tickers", 1, 10, 5)
    run        = st.button("▶️ Run Scan")

if run:
    st.info(f"Fetching target data for {target_sym}…")
    td = fetch_price_data(token_map[target_sym], tgt_start, tgt_end)
    if len(td) < 5:
        st.error("Not enough data for target.")
        st.stop()

    # Plot target
    st.subheader("Target Pattern")
    fig, ax = mpf.plot(td, type='candle', style='charles',
                       title=f"{target_sym} [{tgt_start} → {tgt_end}]",
                       returnfig=True)
    st.pyplot(fig)

    # Extract target features
    tf = extract_features(td)

    # Scan this batch
    patterns, feats = [], []
    st.info("Scanning comparison symbols in this batch…")
    prog = st.progress(0)
    for i, sym in enumerate(symbols, start=1):
        df = fetch_price_data(token_map[sym], comp_start, comp_end)
        if len(df) >= len(td):
            L = len(td)
            for j in range(len(df)-L+1):
                w = df.iloc[j:j+L]
                patterns.append({"Ticker":sym, "Start":w.index[0].date(), "End":w.index[-1].date(), "Data":w})
                feats.append(extract_features(w))
        prog.progress(i/len(symbols))

    feats = np.array(feats)
    sims  = compute_similarity(tf, feats)

    # Attach & sort
    for p, s in zip(patterns, sims):
        p["Similarity"] = s
    patterns.sort(key=lambda x: x["Similarity"], reverse=True)

    # Pick top-K unique (exclude target)
    unique, seen = [], set()
    for p in patterns:
        t = p["Ticker"]
        if t == target_sym or t in seen: continue
        unique.append(p); seen.add(t)
        if len(unique) >= top_k: break

    # Build result table with safe LTP lookup
    rows = []
    for p in unique:
        tkr = p["Ticker"]
        ltp_resp = client.ltpData("NSE", tkr.replace(".NS",""), token_map[tkr])
        # safe extract
        if ltp_resp and isinstance(ltp_resp, dict) and "data" in ltp_resp and "ltp" in ltp_resp["data"]:
            ltp = ltp_resp["data"]["ltp"]
        else:
            ltp = None

        rows.append({
            "Ticker":      tkr,
            "Start":       p["Start"],
            "End":         p["End"],
            "Similarity":  f"{p['Similarity']:.3f}",
            "Current LTP": ltp
        })

    df_res = pd.DataFrame(rows)
    st.subheader(f"Top {len(df_res)} Similar Patterns (Batch {batch_num}/{n_batches})")
    st.dataframe(df_res)

    st.download_button("📥 Download CSV",
        df_res.to_csv(index=False).encode(),
        f"results_batch{batch_num}.csv","text/csv"
    )

    # Plot each unique pattern
    for p in unique:
        st.subheader(f"{p['Ticker']} [{p['Start']} → {p['End']}]  {p['Similarity']:.3f}")
        fig2, ax2 = mpf.plot(p["Data"], type='candle', style='charles', returnfig=True)
        st.pyplot(fig2)
