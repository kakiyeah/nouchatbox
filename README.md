Use it

https://huggingface.co/spaces/ryusenyeah/nouchatbox

| ![Image 1](https://github.com/user-attachments/assets/abee7a6d-70f3-43e5-b610-64e30b1440dc) | ![Image 2](https://github.com/user-attachments/assets/86612f8f-6aac-4640-8e76-25cf6d635959) |
|:--------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|
# 🌾 Agriculture Consultation Chatbot

A friendly and easy-to-understand agricultural consultation chatbot designed for farmers. This chatbot uses prompt engineering to generate responses in a farmer-friendly style, based on DPO dataset principles.

## Features

- **Friendly Conversational Style**: Uses casual, approachable language that farmers can relate to
- **Empathetic Responses**: Shows understanding and empathy for farmers' concerns
- **Concrete Examples**: Includes real experiences and practical examples
- **Clear Expressions**: Avoids technical jargon and uses everyday language

## How It Works

This chatbot uses prompt engineering to guide language models to generate responses in a farmer-friendly style. Instead of fine-tuning models, it leverages carefully crafted system prompts that instruct the model to:

- Use casual, friendly language (e.g., "おじさん", "だよね", "俺")
- Show empathy and understanding (e.g., "そうだよね", "心配だよね", "分かる分かる")
- Provide concrete examples (e.g., "去年参加した○○さん", "実際にやってみると")
- Avoid technical terms and statistical data
- Offer flexible solutions without being pushy

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Access the application at `http://localhost:7860` in your browser.

## Model Selection

The chatbot uses Hugging Face Inference API with automatic fallback to multiple models:

- Primary: `elyza/ELYZA-japanese-Llama-2-7b-fast-instruct`
- Backup models are automatically tried if the primary model is unavailable

You can configure the model by setting the `HF_API_URL` environment variable.

## Response Style

The chatbot generates responses with the following characteristics:

✅ **Recommended Expressions**:
- Empathetic phrases like "そうだよね", "心配だよね"
- Concrete examples like "実際にやってみると", "去年のデータ見ても"
- Friendly expressions like "一緒にやってみようよ", "相談しようね"
- Reassuring words like "大丈夫だよ", "安心して"

❌ **Avoided Expressions**:
- Technical terms like "統計的に見て", "データによれば"
- Formal honorifics like "推奨いたします", "必要となります"
- Excessive use of numbers and percentages

## :pouting_cat: Others

- Built with Gradio for the user interface
- Uses Hugging Face Inference API for model inference
- Implements automatic model fallback for high availability
- Includes a fallback response system for when APIs are unavailable
