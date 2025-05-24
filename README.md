# 🤖 智能文档问答系统 (RAG)

## 项目简介

这是一个基于检索增强生成(RAG)技术的智能文档问答系统，能够从上传的文档中快速检索相关信息并生成准确的回答。系统支持多种文档格式，提供友好的Web界面，具备会话管理、联网搜索等功能。

## ✨ 主要特性

### 📚 多格式文档支持
- **文本格式**: `.txt`, `.md` (Markdown)
- **办公文档**: `.docx` (Word文档), `.xlsx/.xls` (Excel表格)  
- **PDF文档**: `.pdf` 文件
- **智能文档分块**: 自动将长文档切分为语义连贯的块，提高检索精度

### 🔍 智能检索与问答
- **语义向量检索**: 基于M3E中文向量模型，实现语义级文档检索
- **FAISS向量数据库**: 高效的相似性搜索和索引管理
- **流式对话**: 支持实时流式回答生成
- **联网搜索**: 集成SerpAPI，支持实时网络信息检索

### 💬 会话管理
- **多会话支持**: 创建和管理多个独立的对话会话
- **会话历史**: 自动保存对话记录，支持历史会话查看
- **自定义会话名**: 支持编辑和自定义会话标题
- **数据持久化**: 基于SQLite数据库的数据存储

### 🎨 现代化界面
- **响应式设计**: 适配不同屏幕尺寸的设备
- **Vue3 + Element Plus**: 现代化的前端技术栈
- **Markdown渲染**: 支持富文本消息显示
- **文件预览**: 上传前可预览文档内容

## 🛠️ 技术栈

### 后端技术
- **FastAPI**: 现代、快速的Web框架
- **Sentence Transformers**: 语义向量化模型
- **FAISS**: Facebook AI相似性搜索库
- **SQLite**: 轻量级数据库
- **OpenAI API**: 大语言模型接口

### 前端技术
- **Vue 3**: 渐进式JavaScript框架
- **Element Plus**: 基于Vue 3的组件库
- **Marked.js**: Markdown解析渲染

### AI模型
- **M3E模型**: 中文语义向量化模型 (`moka-ai/m3e-base`)
- **OpenAI GPT**: 文本生成和对话

## 📦 安装部署

### 环境要求
- Python 3.8+
- 2GB+ 可用内存
- 互联网连接(首次下载模型)

### 1. 克隆项目
```bash
git clone https://github.com/your-username/rag-document-qa.git
cd rag-document-qa
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 环境配置
创建 `.env` 文件或设置环境变量：
```bash
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，默认官方API

# SerpAPI配置 (可选，用于联网搜索)
SERPAPI_API_KEY=your_serpapi_key_here
```

### 4. 启动服务
```bash
python main.py
```

服务将在 `http://localhost:8000` 启动

## 🚀 使用指南

### 1. 文档管理
- 将需要问答的文档放入 `docs/` 文件夹
- 支持的格式：`.txt`, `.md`, `.docx`, `.pdf`, `.xlsx`, `.xls`
- 系统会自动加载并建立向量索引

### 2. Web界面使用
1. 打开浏览器访问 `http://localhost:8000`
2. 上传或管理文档
3. 创建新的对话会话
4. 输入问题，系统会基于文档内容生成回答

### 3. API接口
系统提供RESTful API接口：

#### 文档管理
- `GET /api/docs` - 获取文档列表
- `POST /api/docs/upload` - 上传文档
- `DELETE /api/docs/{filename}` - 删除文档

#### 对话交互
- `POST /api/chat/stream` - 流式对话
- `GET /api/chat/history` - 获取会话历史
- `DELETE /api/chat/session/{session_id}` - 删除会话

## 📁 项目结构

```
rag-document-qa/
├── main.py                 # 主程序入口
├── file_utils.py          # 文档处理工具
├── requirements.txt       # 项目依赖
├── README.md             # 项目说明
├── docs/                 # 用户上传的文档目录
├── static/               # 静态文件目录
│   └── Index.html       # 前端界面
├── icon/                 # 项目图标资源
├── index/                # 向量数据库索引存储
└── local_m3e_model/      # 本地缓存的向量模型
```

## ⚙️ 配置选项

### 文档分块设置
在 `main.py` 中可调整文档分块参数：
```python
def chunk_document(text, max_chars=500, overlap=100):
    # max_chars: 每个块的最大字符数
    # overlap: 相邻块重叠字符数
```

### 检索设置
调整检索相关度和数量：
```python
def retrieve_docs(query, index, k=5):
    # k: 检索返回的文档块数量
```

## 🔧 故障排除

### 常见问题

**1. 模型下载失败**
- 检查网络连接
- 使用镜像源：设置 `HF_ENDPOINT=https://hf-mirror.com`

**2. 文档加载错误**
- 确认文档格式是否支持
- 检查文档是否损坏
- 查看控制台错误日志

**3. API连接失败**
- 检查OpenAI API密钥是否正确
- 确认API额度是否足够
- 检查网络连接和代理设置

**4. 向量索引问题**
- 删除 `index/` 目录重新建立索引
- 检查磁盘空间是否充足

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 开源协议

本项目基于 [MIT协议](LICENSE) 开源。

## 🙏 致谢

- [Sentence Transformers](https://www.sbert.net/) - 语义向量化
- [FAISS](https://github.com/facebookresearch/faiss) - 向量检索
- [FastAPI](https://fastapi.tiangolo.com/) - Web框架
- [Vue 3](https://vuejs.org/) - 前端框架
- [Element Plus](https://element-plus.org/) - UI组件库

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 [GitHub Issue](https://github.com/Ling-ling123/rag-document-qa/issues)
- 发送邮件至：mangocheesea@gmail.com

---

⭐ 如果这个项目对您有帮助，请给个Star支持一下！ 