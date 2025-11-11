import gradio as gr
import os
import requests

HF_API_URL = os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models/elyza/ELYZA-japanese-Llama-2-7b-fast-instruct")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

BACKUP_MODELS = [
    "https://api-inference.huggingface.co/models/elyza/ELYZA-japanese-Llama-2-7b-instruct",
    "https://api-inference.huggingface.co/models/cyberagent/calm2-7b-chat",
    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    "https://api-inference.huggingface.co/models/google/flan-t5-large",
]

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

def generate_fallback_response(message):
    message_lower = message.lower()
    
    if "リスク" in message or "心配" in message or "不安" in message:
        return "そうだよね、その心配はよく分かるよ。実際にやってみると、最初は不安かもしれないけど、段階的に進めていけば大丈夫だと思うんだ。例えば、一部の田んぼでまず試してみて、効果を自分の目で確かめてから広げるっていう方法もあるよ。一緒に相談しながら進めていこうね。"
    
    elif "収量" in message or "減る" in message or "家族" in message:
        return "家族のこと考えたら、そりゃ心配だよね。だから、もし収量が明らかに減った場合は、補償をするって約束するよ。実際には、去年参加した農家さん、みんな収量は変わってない。むしろ少し増えてるって人もいるんだ。中干しを延ばすと根が深く張るからかもね。でも、万が一のために保険はかけとく。おじさんが損することは絶対にさせない。"
    
    elif "水" in message or "管理" in message or "大変" in message:
        return "確かに、最初は『いつもと違う』って感じるかもしれない。でも実際やってみると、水を入れるタイミングが2週間遅れるだけで、他は今までと全く同じなんだ。むしろ、中干し期間が長いから、その間は水の心配しなくていいって言う人もいるよ。初年度は、俺が週1で様子を見に来て、『今週はこうしよう』ってアドバイスするから、一人で悩まなくて大丈夫。"
    
    elif "高齢" in message or "体力" in message or "覚え" in message:
        return "おじさん、何十年も米作りやってきたベテランじゃないですか。新しいことって言っても、水を抜くタイミングを2週間遅らせるだけ。肥料も変えない、農薬も変えない、機械も変えない。今までのやり方にプラス2週間するだけだよ。体力的にも、特に重労働が増えるわけじゃない。むしろ、中干し期間が長いから、その間は田んぼに行く回数が減るって考えることもできるよ。"
    
    elif "品質" in message or "味" in message or "売れ" in message:
        return "そうだよね、いくら環境に良くても、米が売れなきゃ本末転倒だ。だから今回は、品質を絶対に落とさないってのが大前提なんだ。実は既に参加してる農家さん、去年と同じ等級で出荷できてるし、むしろJAから『環境配慮米』って名前で少し高く買ってもらえたって言ってた。おじさんも直販やってるなら、これを売りにできるかもしれないよ。"
    
    else:
        return "そうだよね、その心配はよく分かるよ。実際にやってみると、最初は不安かもしれないけど、段階的に進めていけば大丈夫だと思うんだ。例えば、一部の田んぼでまず試してみて、効果を自分の目で確かめてから広げるっていう方法もあるよ。一緒に相談しながら進めていこうね。何か具体的な質問があったら、遠慮なく聞いてくれよ。"
    
def format_prompt(user_message, history=None):
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
    prompt = format_prompt(message, history)
    
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    models_to_try = [HF_API_URL] + BACKUP_MODELS
    
    for model_url in models_to_try:
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
            response = requests.post(model_url, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "")
                    generated_text = generated_text.strip()
                    if "[/INST]" in generated_text:
                        generated_text = generated_text.split("[/INST]")[-1].strip()
                    if "<s>" in generated_text:
                        generated_text = generated_text.split("<s>")[-1].strip()
                    return generated_text
                elif isinstance(result, dict) and "generated_text" in result:
                    generated_text = result["generated_text"].strip()
                    if "[/INST]" in generated_text:
                        generated_text = generated_text.split("[/INST]")[-1].strip()
                    return generated_text
                else:
                    continue
            
            elif response.status_code == 503:
                error_info = response.json() if response.content else {}
                estimated_time = error_info.get("estimated_time", 30)
                if model_url == models_to_try[0]:
                    return f"モデルを読み込み中です。約{estimated_time}秒お待ちください。しばらくしてから再度お試しください。"
                else:
                    continue
            
            elif response.status_code in [410, 404]:
                continue
            
            elif response.status_code == 401:
                if not HF_API_TOKEN:
                    continue
                else:
                    continue
            
            else:
                print(f"Model {model_url} returned status {response.status_code}")
                continue
                
        except requests.exceptions.Timeout:
            continue
        except requests.exceptions.RequestException:
            continue
        except Exception:
            continue
    
    return generate_fallback_response(message)

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

