"""駅の相場モデル(デプロイ練習用)。合成データで「駅を選ぶ → model.predict → 結果を表示」を通す。"""

import json
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

# 公開先の作業ディレクトリはリポジトリの根なので、このファイルの位置から辿る
BASE = Path(__file__).parent
ART = BASE / "artifacts"
FEATURES = ["乗降客数", "都心距離_km", "用途区分"]

st.set_page_config(page_title="駅の相場モデル(デプロイ練習用)")


@st.cache_resource
def load_model():
    """モデルを 1 度だけ読み込む。読み込み時の警告(版の違いなど)も返す。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(ART / "model.joblib")
    return model, [f"{w.category.__name__}: {w.message}" for w in caught]


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ART / name)


def same_version(key: str, trained: str, running: str) -> bool:
    """Python はマイナー版まで、ライブラリは完全一致で比べる。"""
    if key == "Python":
        return trained.split(".")[:2] == running.split(".")[:2]
    return trained == running


model, load_warnings = load_model()
features = load_csv("features.csv")
ranking = load_csv("ranking.csv")
golden = load_csv("golden.csv")
meta = json.loads((ART / "meta.json").read_text(encoding="utf-8"))

st.title("駅の相場モデル(デプロイ練習用)")
st.caption("合成データで作った練習用のアプリ。駅名・地価・乗降客数は実在しない。")

if load_warnings:
    st.warning("モデルの読み込みで警告が出た(学習時と実行時の版の違いの可能性)。\n\n" + "\n\n".join(load_warnings))

tab_detail, tab_rank, tab_check = st.tabs(["駅ごとの詳細", "ランキング", "動作確認"])

with tab_detail:
    station = st.selectbox("駅を選ぶ", sorted(features["駅"].unique()))
    rows = features[features["駅"] == station].copy()
    rows["相場"] = model.predict(rows[FEATURES]).round(-2)
    st.dataframe(rows[["地点ID", "用途区分", "地価", "相場"]], hide_index=True)
    st.bar_chart(rows.set_index("地点ID")[["地価", "相場"]], stack=False)

    r = ranking[ranking["駅"] == station].iloc[0]
    col1, col2 = st.columns(2)
    col1.metric("穴場度(残差の平均、対数)", f"{r['穴場度']:+.3f}")
    col2.metric("地点数", int(r["地点数"]))
    if bool(r["参考値"]):
        st.info("地点数が 3 未満なので、穴場度は参考値。")
    st.caption(
        "相場は全データで学習したモデルの予測(円/m²)。"
        "穴場度は、その駅を学習に使わないモデル(GroupKFold)の残差の平均で、マイナスほど相場より安い。"
    )

with tab_rank:
    show = ranking.copy()
    show["参考値"] = show["参考値"].map({True: "参考値", False: ""})
    st.dataframe(show, hide_index=True)

with tab_check:
    st.subheader("ライブラリの版")
    running = {
        "Python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }
    ver = pd.DataFrame(
        {
            "項目": list(running),
            "学習時": [meta["versions"][k] for k in running],
            "実行時": list(running.values()),
        }
    )
    ver["一致"] = [same_version(k, meta["versions"][k], v) for k, v in running.items()]
    st.dataframe(ver, hide_index=True)

    st.subheader("予測値の一致(学習時と実行時)")
    g = golden.copy()
    g["実行時の予測"] = [float(model.predict(features.loc[features["駅"] == s, FEATURES]).mean()) for s in g["駅"]]
    g["一致"] = np.isclose(g["学習時の予測"], g["実行時の予測"], rtol=1e-9, atol=0)
    st.dataframe(g, hide_index=True)
    if g["一致"].all():
        st.success("3 駅の予測値が学習時と一致した。")
    else:
        st.error("予測値が学習時と一致しない。版や特徴量の違いを確かめる。")
    st.caption(
        f"学習日時 {meta['trained_at']} / α={meta['alpha']} / "
        f"RMSE(対数): モデル {meta['rmse_log_model']:.3f}、平均値 {meta['rmse_log_baseline']:.3f}"
    )

st.divider()
st.caption("出典: 合成データ(練習用)。実データを使う場合は、国土数値情報(地価公示・駅別乗降客数)などの出典を書く。")
