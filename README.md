---
title: 農業相談チャットボット
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# 🌾 農業相談チャットボット

農家さん向けの親しみやすく分かりやすい農業相談チャットボットです。DPO（Direct Preference Optimization）データセットに基づいて設計された、農家さんが理解しやすい回答スタイルを実現しています。

## 特徴

- **親しみやすい口語体**: 農家さんが親しみを感じられる話し方
- **共感的な対応**: 農家さんの気持ちに寄り添う回答
- **具体的な例**: 実際の経験や事例を交えた説明
- **分かりやすい表現**: 専門用語を避け、日常的な言葉で説明

## 使用方法

### Hugging Face Spaceでの使用

1. このリポジトリをHugging Face Spaceにアップロード
2. Spaceの設定で以下を指定：
   - **SDK**: Gradio
   - **Hardware**: CPU（小規模モデル）または GPU（大規模モデル）
   - **Environment variables**: 
     - `MODEL_NAME`: 使用するモデル名（例: `elyza/ELYZA-japanese-Llama-2-7b-instruct`）

### ローカルでの実行

```bash
# 依存関係のインストール
pip install -r requirements.txt

# アプリの起動
python app.py
```

ブラウザで `http://localhost:7860` にアクセスしてください。

## モデルの選択

デフォルトでは `elyza/ELYZA-japanese-Llama-2-7b-instruct` を使用しますが、以下のような日本語対応モデルも使用可能です：

- `elyza/ELYZA-japanese-Llama-2-7b-instruct`
- `elyza/ELYZA-japanese-Llama-2-7b-fast-instruct`
- `cyberagent/calm2-7b-chat`
- その他の日本語LLM

環境変数 `MODEL_NAME` で変更できます。

## 回答スタイル

このチャットボットは、以下の特徴を持つ回答を生成します：

✅ **推奨される表現**:
- 「そうだよね」「心配だよね」などの共感的な表現
- 「実際にやってみると」「去年のデータ見ても」などの具体例
- 「一緒にやってみようよ」「相談しようね」などの親しみやすい表現
- 「大丈夫だよ」「安心して」などの安心感を与える言葉

❌ **避ける表現**:
- 「統計的に見て」「データによれば」などの専門的表現
- 「推奨いたします」「必要となります」などの硬い敬語
- 数値やパーセンテージを多用する説明

## カスタマイズ

`app.py` の `SYSTEM_PROMPT` を編集することで、回答スタイルをカスタマイズできます。

## ライセンス

このプロジェクトは、元のDPOデータセットのライセンスに従います。

## 注意事項

- このチャットボットは農業相談の補助ツールとして設計されています
- 重要な判断は、必ず専門家やJA（農業協同組合）に相談してください
- モデルの回答は参考情報として扱い、実際の農業作業では慎重に判断してください

