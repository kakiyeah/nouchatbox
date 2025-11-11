"""
Hugging Face Inference APIを使用する軽量版
このバージョンはモデルをローカルにダウンロードせず、Hugging FaceのInference APIを使用します
"""
import gradio as gr
import os
import requests

# Hugging Face Inference APIの設定
HF_API_URL = os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models/elyza/ELYZA-japanese-Llama-2-7b-instruct")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

# 优化的System Prompt
SYSTEM_PROMPT = """あなたは農業推進事業者です。農家さんの質問や懸念に対して、共感的で具体的な回答をしてください。

回答のスタイル：
- 親しみやすい口語体を使う（「おじさん」「俺」「だよね」など）
- 農家さんの気持ちに共感する（「そうだよね」「心配だよね」「分かる分かる」など）
- 具体的な例や実際の経験を挙げる（「去年参加した○○さん」「実際にやってみると」など）
- 技術的な専門用語や統計データは避ける
- 柔軟な解決策を提案する
- 日常的な言葉や比喩を使う
- 農家さんを尊重し、強制しない姿勢を示す
- 実践的で分かりやすい説明をする

以下のような表現を避ける：
- 「統計的に見て」「データによれば」「科学的根拠に基づき」などの専門的表現
- 「推奨いたします」「必要となります」などの硬い敬語
- 数値やパーセンテージを多用する説明

代わりに、以下のような表現を使う：
- 「実際にやってみると」「去年のデータ見ても」
- 「一緒にやってみようよ」「相談しようね」
- 「大丈夫だよ」「安心して」などの安心感を与える言葉"""

def format_prompt(user_message, history=None):
    """格式化提示词"""
    conversation = ""
    if history:
        for user_msg, assistant_msg in history:
            if assistant_msg:
                conversation += f"ユーザー: {user_msg}\nアシスタント: {assistant_msg}\n\n"
    
    full_message = conversation + f"ユーザー: {user_message}\nアシスタント: "
    
    prompt = f"""<s>[INST] <<SYS>>
{SYSTEM_PROMPT}
<</SYS>>

{full_message} [/INST]"""
    return prompt

def generate_response(message, history):
    """使用Hugging Face Inference API生成回答"""
    prompt = format_prompt(message, history)
    
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False,
        },
        "options": {
            "wait_for_model": True
        }
    }
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get("generated_text", "")
            # 清理回答
            generated_text = generated_text.strip()
            # 移除可能的重复提示词
            if "[/INST]" in generated_text:
                generated_text = generated_text.split("[/INST]")[-1].strip()
            return generated_text
        else:
            return "申し訳ございませんが、回答を生成できませんでした。もう一度お試しください。"
    
    except requests.exceptions.RequestException as e:
        return f"APIリクエストエラー: {str(e)}。もう一度お試しください。"
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

# 创建Gradio界面
def create_interface():
    with gr.Blocks(title="農業相談チャットボット", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🌾 農業相談チャットボット
        
        農家さんの質問や懸念に対して、親しみやすく分かりやすい回答をします。
        農業に関する質問を気軽にどうぞ！
        
        **注意**: このバージョンはHugging Face Inference APIを使用しています。
        """)
        
        chatbot = gr.Chatbot(
            label="チャット",
            height=500,
            show_copy_button=True
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="メッセージ",
                placeholder="農業に関する質問を入力してください...",
                scale=4,
                lines=2
            )
            submit_btn = gr.Button("送信", variant="primary", scale=1)
        
        with gr.Row():
            clear_btn = gr.Button("会話をクリア", variant="secondary")
        
        # 示例问题
        gr.Markdown("### 💡 質問例")
        examples = gr.Examples(
            examples=[
                "中干期を延ばすと米がパサパサになるって聞いたんだけど、そんなリスクは取りたくないんだよ。",
                "収量が減ったらどうするんだ？家族を養っていかなきゃならないんだよ。",
                "水の管理が難しくなるんじゃないか？今でも大変なのに。",
                "高齢で体力に自信がないんだ。新しいことを覚えられるかな。",
            ],
            inputs=msg,
            label="クリックして試してみてください"
        )
        
        # 事件处理
        def user(user_message, history):
            return "", history + [[user_message, None]]
        
        def bot(history):
            if not history or not history[-1][0]:
                return history
            
            user_message = history[-1][0]
            response = generate_response(user_message, history[:-1])
            history[-1][1] = response
            return history
        
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        submit_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear_btn.click(lambda: None, None, chatbot, queue=False)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

