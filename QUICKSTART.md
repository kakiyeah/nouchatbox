# 快速上手指南

## 📋 项目概述

这个项目是一个基于提示词（Prompt）的农业咨询聊天机器人，旨在生成与您提供的JSON数据集中`chosen`回答风格相似的回复。

## 🚀 快速部署到Hugging Face Space

### 步骤1: 创建Hugging Face Space

1. 访问 https://huggingface.co/spaces
2. 点击 "Create new Space"
3. 填写信息：
   - **Space name**: `agriculture-chatbot` (或您喜欢的名称)
   - **SDK**: 选择 **Gradio**
   - **Hardware**: 
     - 如果使用7B模型: 选择 **GPU T4 small** 或 **GPU T4 medium**
     - 如果使用Inference API: 选择 **CPU basic**

### 步骤2: 上传文件

将以下文件上传到Space：

- ✅ `app.py` - 主应用文件（使用本地模型）
- ✅ `requirements.txt` - Python依赖
- ✅ `README.md` - 说明文档
- ✅ `system_prompt.txt` - 系统提示词（可选）

**或者**使用Inference API版本：

- ✅ `app_hf_inference.py` - 重命名为 `app.py`（使用Hugging Face Inference API）

### 步骤3: 配置环境变量（可选）

在Space的 **Settings > Variables and secrets** 中添加：

- `MODEL_NAME`: 模型名称（默认: `elyza/ELYZA-japanese-Llama-2-7b-instruct`）

如果使用Inference API版本：

- `HF_API_URL`: API URL
- `HF_API_TOKEN`: 您的Hugging Face token（可选，用于提高速率限制）

### 步骤4: 等待部署

Space会自动构建和部署。查看日志确认没有错误。

## 📝 文件说明

### 主要文件

- **app.py**: 主应用，直接加载模型到本地
- **app_hf_inference.py**: 使用Hugging Face Inference API的轻量版本
- **requirements.txt**: Python依赖包
- **system_prompt.txt**: 系统提示词模板

### 配置文件

- **config.json**: 模型配置（当前未使用，可扩展）
- **.gitignore**: Git忽略文件

### 文档

- **README.md**: 项目说明（日语）
- **README_CN.md**: 项目说明（中文）
- **DEPLOYMENT.md**: 详细部署指南
- **QUICKSTART.md**: 本文件

## 🎯 回答风格说明

基于您提供的JSON数据，系统提示词已优化为生成以下风格的回答：

### ✅ 推荐风格（chosen回答的特点）

1. **口语化、亲切**
   - 使用「おじさん」「俺」「だよね」等
   - 例如: "おじさんが心配しているその点は本当に大事で..."

2. **共情和理解**
   - 使用「そうだよね」「心配だよね」「分かる分かる」
   - 例如: "そうだよね、いくら環境に良くても、米が売れなきゃ本末転倒だ"

3. **具体例子**
   - 引用实际经验: "去年参加した15軒、みんな収量は変わってない"
   - 提到具体人物: "鈴木さんとこ、去年と同じ等級で出荷できてる"

4. **灵活解决方案**
   - 提供选择: "まずは見学だけでもどう？"
   - 不强求: "強制じゃないから"

### ❌ 避免风格（rejected回答的特点）

1. **技术术语**
   - 避免: "統計的に見て品質劣化の確率は0.3%以下"
   - 避免: "AG-005方法論の技術仕様書によれば"

2. **正式敬语**
   - 避免: "推奨いたします"
   - 避免: "必要となります"

3. **统计数据**
   - 避免: "97.2%を占めており"
   - 避免: "±2%の範囲内であり"

## 🔧 自定义提示词

编辑 `app.py` 中的 `SYSTEM_PROMPT` 变量来自定义回答风格。

当前提示词已根据您的JSON数据优化，包含：
- 风格指导
- 推荐表达
- 避免表达
- 具体示例

## 🧪 测试

部署后，可以尝试以下示例问题：

1. "中干期を延ばすと米がパサパサになるって聞いたんだけど、そんなリスクは取りたくないんだよ。"
2. "収量が減ったらどうするんだ？家族を養っていかなきゃならないんだよ。"
3. "水の管理が難しくなるんじゃないか？今でも大変なのに。"

回答应该：
- ✅ 使用口语化、亲切的语言
- ✅ 显示理解和共情
- ✅ 提供具体例子
- ✅ 避免技术术语

## 📊 性能优化建议

1. **使用更小的模型**: 如果响应慢，考虑使用更小的模型
2. **使用Inference API**: `app_hf_inference.py` 版本不需要本地加载模型
3. **调整参数**: 在 `generate_response` 函数中调整 `max_new_tokens`、`temperature` 等

## 🐛 常见问题

### Q: 模型加载失败
A: 检查Hardware设置是否正确，确保有足够的GPU/CPU资源

### Q: 回答风格不符合预期
A: 调整 `SYSTEM_PROMPT`，参考JSON数据中的`chosen`回答

### Q: 响应太慢
A: 使用更小的模型或Inference API版本

### Q: 内存不足
A: 使用Inference API版本（`app_hf_inference.py`）或升级Hardware

## 📚 下一步

1. 测试不同的模型
2. 根据实际使用情况调整提示词
3. 收集用户反馈并优化
4. 考虑添加更多功能（如历史记录、导出等）

## 💡 提示

- 提示词的质量直接影响回答风格
- 定期根据用户反馈调整提示词
- 可以A/B测试不同的提示词版本

