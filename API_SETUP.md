# API 配置指南

## 为什么需要配置 API？

Hugging Face 的免费 Inference API 有速率限制，某些模型可能需要认证 token 才能使用。配置 token 可以提高：
- API 调用成功率
- 响应速度
- 可用模型范围

## 方法1: 添加 Hugging Face Access Token（推荐）

### 步骤1: 创建 Access Token

1. 访问：https://huggingface.co/settings/tokens
2. 点击 "New token"
3. 填写信息：
   - **Name**: `space-api-token`（或任意名称）
   - **Type**: 选择 **Write**（需要写入权限才能使用某些模型）
4. 点击 "Generate a token"
5. **重要**：复制生成的 token（只显示一次！）

### 步骤2: 在 Space 中添加 Token

1. 进入你的 Space 页面
2. 点击 "Settings"（设置）
3. 找到 "Variables and secrets" 部分
4. 点击 "New secret"
5. 添加：
   - **Key**: `HF_API_TOKEN`
   - **Value**: 粘贴你刚才复制的 token
6. 点击 "Add secret"
7. Space 会自动重新构建

### 步骤3: 测试

等待 Space 重新构建完成后，测试聊天功能。如果配置正确，API 调用应该会成功。

## 方法2: 使用其他模型

如果默认模型不可用，可以在 Space 设置中指定其他模型：

### 在 Space 设置中添加环境变量

1. 进入 Space Settings
2. 找到 "Variables and secrets"
3. 添加新的环境变量：
   - **Key**: `HF_API_URL`
   - **Value**: 选择以下之一：
     - `https://api-inference.huggingface.co/models/elyza/ELYZA-japanese-Llama-2-7b-instruct`
     - `https://api-inference.huggingface.co/models/cyberagent/calm2-7b-chat`
     - `https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2`

## 方法3: 使用 OpenAI API（可选）

如果你有 OpenAI API key，也可以使用 OpenAI 的模型。需要修改代码，我可以帮你实现。

## 当前代码的自动重试机制

代码已经实现了自动重试多个模型的功能：

1. **主模型**: `elyza/ELYZA-japanese-Llama-2-7b-fast-instruct`
2. **备用模型1**: `elyza/ELYZA-japanese-Llama-2-7b-instruct`
3. **备用模型2**: `cyberagent/calm2-7b-chat`
4. **备用模型3**: `mistralai/Mistral-7B-Instruct-v0.2`

如果主模型失败，会自动尝试备用模型。

## 检查 API 状态

### 查看 Logs

在 Space 的 "Logs" 标签中，你可以看到：
- 哪些模型被尝试了
- 每个模型的响应状态码
- 错误信息（如果有）

### 常见状态码

- **200**: 成功 ✅
- **503**: 模型正在加载中，需要等待
- **410**: 模型端点不可用（已移除）
- **404**: 模型不存在
- **401**: 需要认证 token
- **429**: 速率限制（请求太频繁）

## 故障排除

### 问题1: 所有模型都返回 410 或 404

**解决方案**:
- 检查模型名称是否正确
- 尝试使用其他模型
- 添加 `HF_API_TOKEN` 可能有助于访问某些模型

### 问题2: 返回 503（模型加载中）

**解决方案**:
- 这是正常的，模型需要时间加载
- 等待提示的时间后重试
- 或者代码会自动尝试下一个模型

### 问题3: 返回 401（需要认证）

**解决方案**:
- 添加 `HF_API_TOKEN` 环境变量
- 确保 token 有正确的权限

### 问题4: 返回 429（速率限制）

**解决方案**:
- 等待一段时间后重试
- 添加 `HF_API_TOKEN` 可以提高速率限制
- 考虑使用付费 API 计划

## 测试 API 是否工作

你可以手动测试 API：

```bash
curl https://api-inference.huggingface.co/models/elyza/ELYZA-japanese-Llama-2-7b-fast-instruct \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "こんにちは"}'
```

## 推荐配置

为了获得最佳体验，建议：

1. ✅ 添加 `HF_API_TOKEN`（提高成功率和速率限制）
2. ✅ 使用默认的主模型（已经过优化）
3. ✅ 让代码自动尝试备用模型（已实现）

## 需要帮助？

如果遇到问题：
1. 查看 Space 的 Logs 标签
2. 检查环境变量是否正确设置
3. 确认 token 是否有效
4. 尝试不同的模型

