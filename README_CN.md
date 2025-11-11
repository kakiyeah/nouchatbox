# 🌾 农业咨询聊天机器人

面向农民的友好易懂的农业咨询聊天机器人。基于DPO（Direct Preference Optimization）数据集设计，实现农民易于理解的回答风格。

## 特点

- **友好的口语化表达**: 让农民感到亲切的说话方式
- **共情式回应**: 理解农民心情的回答
- **具体实例**: 结合实际经验和案例的说明
- **易懂的表达**: 避免专业术语，使用日常用语说明

## 使用方法

### 在Hugging Face Space上使用

1. 将此仓库上传到Hugging Face Space
2. 在Space设置中指定：
   - **SDK**: Gradio
   - **Hardware**: CPU（小模型）或 GPU（大模型）
   - **Environment variables**: 
     - `MODEL_NAME`: 要使用的模型名称（例如: `elyza/ELYZA-japanese-Llama-2-7b-instruct`）

### 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 启动应用
python app.py
```

在浏览器中访问 `http://localhost:7860`。

## 模型选择

默认使用 `elyza/ELYZA-japanese-Llama-2-7b-instruct`，但也可以使用以下日语模型：

- `elyza/ELYZA-japanese-Llama-2-7b-instruct`
- `elyza/ELYZA-japanese-Llama-2-7b-fast-instruct`
- `cyberagent/calm2-7b-chat`
- 其他日语LLM

可以通过环境变量 `MODEL_NAME` 更改。

## 回答风格

此聊天机器人生成具有以下特点的回答：

✅ **推荐的表达**:
- 「そうだよね」「心配だよね」等共情表达
- 「実際にやってみると」「去年のデータ見ても」等具体例子
- 「一緒にやってみようよ」「相談しようね」等友好表达
- 「大丈夫だよ」「安心して」等让人安心的词语

❌ **避免的表达**:
- 「統計的に見て」「データによれば」等专业表达
- 「推奨いたします」「必要となります」等生硬敬语
- 大量使用数字和百分比的说明

## 自定义

编辑 `app.py` 中的 `SYSTEM_PROMPT` 可以自定义回答风格。

## 许可证

本项目遵循原始DPO数据集的许可证。

## 注意事项

- 此聊天机器人设计为农业咨询的辅助工具
- 重要决策请务必咨询专家或JA（农业协同组合）
- 请将模型的回答作为参考信息，在实际农业作业中请谨慎判断

