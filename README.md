# 駅の相場モデル(デプロイ練習用)

合成データで「学習 → joblib で保存 → Streamlit Community Cloud で公開 → 駅を選ぶ → model.predict → 結果を表示」を一通り通すための練習用一式。駅名・地価・乗降客数は実在しない。

## 中身

| ファイル | 役割 |
|---|---|
| train.py | Colab の代わり。合成データで学習し、artifacts/ に成果物を書き出す |
| artifacts/ | model.joblib(前処理と目的変数の対数変換を含むモデル)・features.csv(predict に渡す特徴量)・ranking.csv(穴場度)・golden.csv(3 駅の予測値)・meta.json(学習時の版) |
| app.py | 公開するアプリ。「動作確認」タブで、学習時と実行時の版と、3 駅の予測値の一致を表示する |
| requirements.txt | 公開先が読む依存ファイル。版を == で固定している |
| test_app.py | アプリを画面なしで動かす確認(streamlit.testing の AppTest) |
| check_version_mismatch.py | 学習時と違う版の環境でモデルを読む実験 |

## 手元で動かす(公開先の代わり)

Python 3.12 の環境を requirements.txt だけから作り、そこで動かす。

```bash
uv venv --python 3.12 .venv-cloud
uv pip install --python .venv-cloud -r requirements.txt
.venv-cloud/bin/streamlit run app.py --browser.gatherUsageStats false
```

## Streamlit Community Cloud に公開する

画面の文言は公式文書(2026-11-22 時点)の記述による。変わっている可能性がある。

1. GitHub で**公開**リポジトリを作り、次のファイルを置く: app.py・requirements.txt・artifacts/ の 5 ファイル・README.md(train.py などは置いてもよい)。.venv-train/ と .venv-cloud/ は置かない。
2. share.streamlit.io に GitHub アカウントでログインし、GitHub との連携を許可する。
3. 「Create app」から GitHub のリポジトリを選ぶ。ブランチと、メインのファイル(app.py)を指定する。
4. 「Advanced settings」で Python の版を **3.12** にする(学習時と同じ版。公開後に変えるには、アプリを削除して公開し直す必要がある)。
5. 「Deploy」を押し、ログで導入の進み具合を見る(公開後のログは、アプリ右下の「Manage app」から見られる)。
6. 公開されたら、ログアウトした別の端末かブラウザで URL を開き、「動作確認」タブで版と予測値がすべて一致していることを見る。

## 片付け

練習が済んだら、アプリを Community Cloud から削除し、必要ならリポジトリも削除する。アプリは 12 時間アクセスが無いと休止する。
