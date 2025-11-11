import gradio as gr
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# 加载模型和tokenizer
MODEL_NAME = os.getenv("MODEL_NAME", "elyza/ELYZA-japanese-Llama-2-7b-instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化模型和tokenizer
tokenizer = None
model = None
pipe = None

def load_model():
    """加载模型"""
    global tokenizer, model, pipe
    
    if tokenizer is not None and model is not None:
        return
    
    try:
        print(f"正在加载模型: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # 尝试使用pipeline（更简单）
        try:
            pipe = pipeline(
                "text-generation",
                model=MODEL_NAME,
                tokenizer=MODEL_NAME,
                device=0 if device == "cuda" else -1,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            print("使用pipeline加载模型成功")
        except:
            # 如果pipeline失败，使用传统方法
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
            )
            if device == "cpu":
                model = model.to(device)
            print("使用传统方法加载模型成功")
            
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("将使用模拟模式（仅用于测试）")
        tokenizer = None
        model = None
        pipe = None

# 在启动时加载模型
load_model()

# 优化的System Prompt - 基于JSON数据中的chosen回答风格
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

def format_prompt(user_message):
    """格式化提示词"""
    prompt = f"""<s>[INST] <<SYS>>
{SYSTEM_PROMPT}
<</SYS>>

{user_message} [/INST]"""
    return prompt

def generate_response(message, history):
    """生成回答"""
    # 确保模型已加载
    if tokenizer is None:
        load_model()
    
    if model is None and pipe is None and tokenizer is None:
        # 模拟模式 - 返回示例回答（基于JSON数据中的风格）
        return "そうだよね、その心配はよく分かるよ。実際にやってみると、最初は不安かもしれないけど、段階的に進めていけば大丈夫だと思うんだ。例えば、一部の田んぼでまず試してみて、効果を自分の目で確かめてから広げるっていう方法もあるよ。一緒に相談しながら進めていこうね。"
    
    # 构建完整的对话历史
    conversation = ""
    if history:
        for user_msg, assistant_msg in history:
            conversation += f"ユーザー: {user_msg}\nアシスタント: {assistant_msg}\n\n"
    
    full_message = conversation + f"ユーザー: {message}\nアシスタント: "
    
    # 格式化提示词
    prompt = format_prompt(full_message)
    
    try:
        # 使用pipeline（如果可用）
        if pipe is not None:
            outputs = pipe(
                prompt,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                return_full_text=False,
            )
            response = outputs[0]["generated_text"].strip()
            return response
        
        # 使用传统方法
        elif model is not None and tokenizer is not None:
            # 编码输入
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            if device == "cuda":
                inputs = inputs.to(device)
            
            # 生成回答
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
                )
            
            # 解码输出
            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            response = response.strip()
            return response
        
        else:
            return "モデルの読み込みに失敗しました。Hugging Face Spaceの設定を確認してください。"
    
    except Exception as e:
        return f"エラーが発生しました: {str(e)}。もう一度お試しください。"

# 创建Gradio界面
def create_interface():
    with gr.Blocks(title="農業相談チャットボット", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🌾 農業相談チャットボット
        
        農家さんの質問や懸念に対して、親しみやすく分かりやすい回答をします。
        農業に関する質問を気軽にどうぞ！
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

