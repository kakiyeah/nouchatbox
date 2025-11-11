# Hugging Face Space デプロイガイド

このプロジェクトをHugging Face Spaceにデプロイする方法を説明します。

## 方法1: モデルを直接ロードする方法（app.py）

### 手順

1. **Hugging Face Spaceを作成**
   - https://huggingface.co/spaces にアクセス
   - "Create new Space" をクリック
   - Space名を入力（例: `agriculture-chatbot`）
   - SDK: **Gradio** を選択
   - Hardware: モデルサイズに応じて選択
     - 7Bモデル: **GPU T4 small** または **GPU T4 medium**
     - より小さいモデル: **CPU basic**

2. **ファイルをアップロード**
   - 以下のファイルをSpaceにアップロード：
     - `app.py`
     - `requirements.txt`
     - `README.md`
     - `system_prompt.txt`（オプション）

3. **環境変数の設定（オプション）**
   - SpaceのSettings > Variables and secrets
   - `MODEL_NAME`: 使用するモデル名（デフォルト: `elyza/ELYZA-japanese-Llama-2-7b-instruct`）

4. **デプロイ**
   - Spaceが自動的にビルドとデプロイを開始します
   - ログを確認して、エラーがないか確認してください

### 推奨モデル

- `elyza/ELYZA-japanese-Llama-2-7b-instruct` - 日本語に最適化されたモデル
- `elyza/ELYZA-japanese-Llama-2-7b-fast-instruct` - より高速なバージョン
- `cyberagent/calm2-7b-chat` - 軽量で高速

## 方法2: Inference APIを使用する方法（app_hf_inference.py）

この方法はモデルをローカルにダウンロードしないため、より軽量です。

### 手順

1. **Hugging Face Spaceを作成**
   - SDK: **Gradio** を選択
   - Hardware: **CPU basic** で十分

2. **ファイルをアップロード**
   - `app_hf_inference.py` を `app.py` にリネーム
   - または、Spaceの設定でエントリーポイントを変更

3. **環境変数の設定**
   - `HF_API_URL`: Inference APIのURL（例: `https://api-inference.huggingface.co/models/elyza/ELYZA-japanese-Llama-2-7b-instruct`）
   - `HF_API_TOKEN`: Hugging Faceのアクセストークン（オプション、レート制限を緩和）

4. **デプロイ**

### 注意事項

- Inference APIは無料プランではレート制限があります
- 有料プランを使用する場合は、`HF_API_TOKEN` を設定してください

## トラブルシューティング

### モデルの読み込みに失敗する

- Hardware設定を確認（GPUが必要な場合）
- モデル名が正しいか確認
- ログを確認してエラーメッセージを確認

### メモリ不足エラー

- より小さいモデルを使用
- Hardwareをアップグレード
- `app_hf_inference.py` を使用（Inference API）

### 応答が遅い

- GPU Hardwareを使用
- より高速なモデルを使用（例: `fast-instruct` バージョン）
- `max_new_tokens` を減らす

## カスタマイズ

### モデルの変更

`app.py` の `MODEL_NAME` を変更するか、環境変数で設定：

```python
MODEL_NAME = os.getenv("MODEL_NAME", "your-model-name")
```

### プロンプトの調整

`app.py` の `SYSTEM_PROMPT` を編集して、回答スタイルを調整できます。

### UIのカスタマイズ

`create_interface()` 関数内のGradioコンポーネントを編集して、UIをカスタマイズできます。

## パフォーマンス最適化

1. **モデルの量子化**: 8bitや4bit量子化を使用してメモリ使用量を削減
2. **バッチ処理**: 複数のリクエストをバッチ処理
3. **キャッシング**: よく使われるプロンプトをキャッシュ

## セキュリティ

- 環境変数に機密情報を保存
- 入力の検証とサニタイゼーション
- レート制限の実装（必要に応じて）

