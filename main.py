
import os
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
import re
from file_utils import load_documents_from_directory
from fastapi import FastAPI, UploadFile, File, Form, Request, Query
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
import json
import shutil
from uuid import uuid4
from datetime import datetime
import sqlite3
import pickle
import asyncio
import requests
import html

# 获取SerpAPI密钥
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')

# 初始化数据库
def init_db():
    """初始化SQLite数据库，创建必要的表"""
    conn = sqlite3.connect('tax_assistant.db')
    cursor = conn.cursor()
    
    # 创建文档表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 创建会话表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id TEXT PRIMARY KEY,
        summary TEXT NOT NULL,
        custom_summary BOOLEAN DEFAULT 0,
        timestamp TEXT NOT NULL
    )
    ''')
    
    # 创建消息表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
    )
    ''')
    
    # 创建向量索引表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS vector_index (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        index_name TEXT NOT NULL,
        index_data BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 存储映射信息的表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS vector_mappings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mapping_name TEXT NOT NULL,
        mapping_data BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    print("数据库初始化完成")

# 调用初始化数据库
init_db()

# 加载文档
documents, sources, errors = load_documents_from_directory("./docs")

# 显示结果
print(f"\n成功加载 {len(documents)} 个文档:")
for i, (doc, source) in enumerate(zip(documents, sources)):
    print(f"\n文档 {i+1} 来源: {source}")
    # 只显示文档的开头部分
    preview = doc[:200] + "..." if len(doc) > 200 else doc
    print(f"内容预览: {preview}")

# 如果有错误，显示错误信息
if errors:
    print("\n加载过程中出现以下错误:")
    for error in errors:
        print(f"- {error}")

# 加载 M3E 模型（先检查本地是否存在）
local_model_path = 'local_m3e_model'
if os.path.exists(local_model_path):
    print(f"从本地加载模型: {local_model_path}")
    model = SentenceTransformer(local_model_path)
else:
    print(f"本地模型不存在，从网络加载: moka-ai/m3e-base")
    model = SentenceTransformer('moka-ai/m3e-base')
    # 保存到本地，以便下次使用
    print(f"保存模型到本地: {local_model_path}")
    model.save(local_model_path)

# 打印加载成功
print("模型加载成功！\n")

# 文档分块函数
def chunk_document(text, max_chars=500, overlap=100):
    """
    将长文档切分成较小的块，使用滑动窗口确保上下文连贯性

    参数:
        text: 要切分的文本
        max_chars: 每个块的最大字符数
        overlap: 相邻块之间的重叠字符数

    返回:
        chunks: 切分后的文本块列表
    """
    # 如果文本长度小于最大长度，直接返回
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # 确定当前块的结束位置
        end = start + max_chars

        # 如果没有到达文本末尾，尝试在句子边界切分
        if end < len(text):
            # 在结束位置查找最近的句子结束标记
            sentence_ends = [
                m.end() for m in re.finditer(r'[。！？.!?]\s*', text[start:end])
            ]

            if sentence_ends:  # 如果找到句子结束标记，在最后一个句子结束处切分
                end = start + sentence_ends[-1]
            else:  # 如果没有找到，尝试在单词或标点处切分
                last_space = text[start:end].rfind(' ')
                last_punct = max(text[start:end].rfind('，'), text[start:end].rfind(','))
                cut_point = max(last_space, last_punct)

                if cut_point > 0:  # 如果找到了合适的切分点
                    end = start + cut_point + 1

        # 添加当前块到结果列表
        chunks.append(text[start:end])

        # 移动开始位置，考虑重叠
        start = end - overlap

        # 确保开始位置不会后退
        if start < 0:
            start = 0

        # 避免无限循环
        if start >= len(text):
            break

    return chunks

def get_embeddings(texts):
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.array(embeddings)

# 用于从数据库加载或保存向量索引和映射
def load_vector_index_from_db():
    """从数据库加载向量索引和映射数据"""
    conn = sqlite3.connect('tax_assistant.db')
    cursor = conn.cursor()
    
    # 检查向量索引是否存在
    cursor.execute("SELECT index_data FROM vector_index WHERE index_name = 'faiss_index' ORDER BY created_at DESC LIMIT 1")
    index_row = cursor.fetchone()
    
    # 检查映射数据是否存在
    cursor.execute("SELECT mapping_data FROM vector_mappings WHERE mapping_name = 'chunks_mapping' ORDER BY created_at DESC LIMIT 1")
    mapping_row = cursor.fetchone()
    
    if index_row and mapping_row:
        print("从数据库加载向量索引和映射数据")
        try:
            # 从二进制数据恢复FAISS索引
            index_binary = index_row[0]
            # 将bytes对象转换为numpy数组，这是faiss.deserialize_index所需的格式
            index_binary_np = np.frombuffer(index_binary, dtype=np.uint8)
            index = faiss.deserialize_index(index_binary_np)
            
            # 从二进制数据恢复映射
            mapping_binary = mapping_row[0]
            mapping_data = pickle.loads(mapping_binary)
            
            conn.close()
            return index, mapping_data
        except Exception as e:
            print(f"加载向量索引时出错: {e}")
            conn.close()
            return None, None
    
    conn.close()
    return None, None

def save_vector_index_to_db(index, mapping_data):
    """将向量索引和映射数据保存到数据库"""
    conn = sqlite3.connect('tax_assistant.db')
    cursor = conn.cursor()
    
    try:
        # 序列化FAISS索引
        index_binary = faiss.serialize_index(index)
        
        # 序列化映射数据
        mapping_binary = pickle.dumps(mapping_data)
        
        # 存储索引数据
        cursor.execute("INSERT INTO vector_index (index_name, index_data) VALUES (?, ?)",
                      ('faiss_index', index_binary))
        
        # 存储映射数据
        cursor.execute("INSERT INTO vector_mappings (mapping_name, mapping_data) VALUES (?, ?)",
                      ('chunks_mapping', mapping_binary))
        
        conn.commit()
        conn.close()
        print("向量索引和映射数据已保存到数据库")
    except Exception as e:
        print(f"保存向量索引时出错: {e}")
        conn.close()

# 文档和chunk的映射关系
document_to_chunks = {}
chunks_to_document = {}
all_chunks = []

# 从数据库加载向量索引和映射
index, mapping_data = load_vector_index_from_db()

if index is not None and mapping_data is not None:
    # 使用数据库中的数据
    document_to_chunks = mapping_data['doc_to_chunks']
    chunks_to_document = mapping_data['chunks_to_doc']
    all_chunks = mapping_data['all_chunks']
    print("成功从数据库加载向量索引和映射数据")
else:
    print("数据库中没有找到向量索引，创建新索引")
    # 处理文档并分块
    for doc_id, doc in enumerate(documents):
        # 对长文档进行分块
        chunks = chunk_document(doc)

        # 存储映射关系
        document_to_chunks[doc_id] = []
        for chunk in chunks:
            chunk_id = len(all_chunks)
            all_chunks.append(chunk)
            document_to_chunks[doc_id].append(chunk_id)
            chunks_to_document[chunk_id] = doc_id

    # 生成文档块嵌入
    chunk_embeddings = get_embeddings(all_chunks)

    # 初始化 FAISS 索引
    dimension = chunk_embeddings.shape[1]  # 768 for m3e-base
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)

    # 保存索引和映射到数据库
    mapping_data = {
        'doc_to_chunks': document_to_chunks,
        'chunks_to_doc': chunks_to_document,
        'all_chunks': all_chunks
    }
    save_vector_index_to_db(index, mapping_data)
    print("向量索引和映射已创建并保存到数据库")

# 从向量数据库index中获取相关文档
def retrieve_docs(query, index, k=5):
    """
    检索最相关的文档

    参数:
        query: 查询文本
        index: FAISS索引
        k: 返回的相关chunk数量

    返回:
        按相关性排序的原始文档列表
    """
    query_embedding = get_embeddings([query])
    distances, chunk_indices = index.search(query_embedding, k)

    # 获取包含这些chunks的原始文档ID
    retrieved_doc_ids = set()
    retrieved_chunks = []

    for chunk_idx in chunk_indices[0]:
        if chunk_idx >= 0 and chunk_idx < len(all_chunks):  # 确保索引有效
            doc_id = chunks_to_document.get(int(chunk_idx))
            if doc_id is not None:
                retrieved_doc_ids.add(doc_id)
                retrieved_chunks.append((doc_id, all_chunks[int(chunk_idx)]))

    # 获取原始文档
    retrieved_docs = [documents[doc_id] for doc_id in retrieved_doc_ids]

    # 返回文档和对应的相关块
    return retrieved_docs, retrieved_chunks

# LLM 生成回答
client = OpenAI(
    api_key="your_openai_api_key_here",
    base_url="https://api.openai.com/v1"
)

def generate_answer(query, retrieved_docs, retrieved_chunks):
    """
    基于检索到的文档生成回答

    参数:
        query: 用户查询
        retrieved_docs: 检索到的完整文档
        retrieved_chunks: 检索到的文档块

    返回:
        生成的回答
    """
    # 构建上下文，包含原始文档和相关块
    context = "原始文档:\n" + "\n".join(retrieved_docs)

    # 添加相关块信息
    context += "\n\n相关文本块:\n"
    for doc_id, chunk in retrieved_chunks:
        context += f"[文档{doc_id}] {chunk}\n"

    prompt = f"上下文信息:\n{context}\n\n问题: {query}\n请基于上下文信息回答问题:"

    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {"role": "system",
             "content": "你是一个专业的问答助手。请仅基于提供的上下文信息回答问题，不要添加任何未在上下文中提及的信息。"},
            {"role": "user", "content": prompt}
        ],
        # max_tokens=150
    )
    return response.choices[0].message.content

    
app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
app.mount("/icon", StaticFiles(directory="icon"), name="icon")

# 数据库操作函数
def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect('tax_assistant.db')
    conn.row_factory = sqlite3.Row  # 允许通过列名访问
    return conn

@app.get("/api/docs")
def list_docs():
    """列出所有文档文件名"""
    files = os.listdir("./docs")
    return {"files": files}

@app.post("/api/docs/preview")
async def preview_doc(file: UploadFile = File(...)):
    """预览文档内容，返回前2000个字符"""
    try:
        content = await file.read()
        # 假设文件是UTF-8编码的文本文件
        text = content.decode('utf-8', errors='ignore')
        # 截取前2000个字符作为预览
        preview = text[:2000]
        # 如果文本被截断，添加提示
        if len(text) > 2000:
            preview += "\n...(内容已截断，仅显示前2000个字符)"
        return {"content": preview, "filename": file.filename}
    except Exception as e:
        return {"error": f"预览失败: {str(e)}"}

@app.post("/api/docs/upload")
def upload_doc(file: UploadFile = File(...)):
    """上传文档"""
    try:
        print(f"[上传] 收到上传请求: {file.filename}")
        file_path = os.path.join("./docs", file.filename)
        print(f"[上传] 保存文件到: {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[上传] 文件保存成功: {file_path}")
        # 将文档保存到数据库
        conn = get_db_connection()
        cursor = conn.cursor()
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        cursor.execute("INSERT INTO documents (filename, content) VALUES (?, ?)",
                      (file.filename, content))
        conn.commit()
        conn.close()
        print(f"[上传] 数据库写入成功: {file.filename}")
        # 上传文档后重建分块和向量数据库
        rebuild_vector_index()
        print(f"[上传] 向量索引重建已触发")
        return {"message": "上传成功", "filename": file.filename}
    except Exception as e:
        print(f"[上传] 失败: {str(e)}")
        return {"error": f"上传失败: {str(e)}"}

@app.delete("/api/docs/{filename}")
def delete_doc(filename: str):
    """删除文档"""
    try:
        file_path = os.path.join("./docs", filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            
            # 从数据库中删除文档
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE filename = ?", (filename,))
            conn.commit()
            conn.close()
            
            # 删除文档后重建分块和向量数据库
            rebuild_vector_index()
            
            return {"message": "删除成功"}
        else:
            return {"message": "文件不存在"}
    except Exception as e:
        return {"error": f"删除失败: {str(e)}"}

def save_message(session_id, role, content, is_new_message=False):
    """
    保存消息到数据库
    
    参数:
        session_id: 会话ID
        role: 消息角色(user/assistant)
        content: 消息内容
        is_new_message: 是否为新消息(如果是，则更新时间戳)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 检查会话是否存在
    cursor.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,))
    session = cursor.fetchone()
    
    # 使用标准ISO格式的时间戳，确保SQLite可以正确解析排序
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not session:
        # 创建新会话，初始显示为"新会话"
        cursor.execute(
            "INSERT INTO chat_sessions (id, summary, custom_summary, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, "新会话", 0, timestamp)
        )
    
    # 添加消息
    cursor.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, timestamp)
    )
    
    # 只有在是新消息时才更新会话时间戳
    if is_new_message:
        cursor.execute(
            "UPDATE chat_sessions SET timestamp = ? WHERE id = ?",
            (timestamp, session_id)
        )
    
    # 如果是用户的第一条消息且没有自定义标题，使用它作为摘要
    if role == "user":
        cursor.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'user'", 
            (session_id,)
        )
        user_msg_count = cursor.fetchone()[0]
        
        if user_msg_count == 1:
            cursor.execute("SELECT custom_summary FROM chat_sessions WHERE id = ?", (session_id,))
            custom_summary = cursor.fetchone()[0]
            
            if not custom_summary:
                summary = content[:20] + ("..." if len(content) > 20 else "")
                cursor.execute(
                    "UPDATE chat_sessions SET summary = ? WHERE id = ?",
                    (summary, session_id)
                )
    
    conn.commit()
    conn.close()
    
    return session_id

@app.get("/api/chat/history")
def get_chat_history():
    """获取所有历史会话摘要，按时间戳倒序排列"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 使用 datetime 函数将字符串时间戳转换为日期时间进行排序
    cursor.execute("SELECT id, summary, timestamp FROM chat_sessions ORDER BY datetime(timestamp) DESC")
    sessions = cursor.fetchall()
    
    result = [
        {
            "id": session["id"],
            "summary": session["summary"],
            "timestamp": session["timestamp"]
        }
        for session in sessions
    ]
    
    conn.close()
    
    # 添加调试日志
    print("获取会话历史，当前会话数:", len(result))
    for i, session in enumerate(result):
        print(f"会话 {i+1}: ID: {session['id']}, 摘要: {session['summary']}, 时间: {session['timestamp']}")
    
    return result

@app.get("/api/chat/session/{session_id}")
def get_chat_session(session_id: str, update_timestamp: bool = False):
    """获取单个会话的全部消息"""
    conn = get_db_connection()
    cursor = conn.cursor()
    # 获取会话消息
    cursor.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id", (session_id,))
    messages = cursor.fetchall()
    # 获取会话信息
    cursor.execute("SELECT summary, timestamp FROM chat_sessions WHERE id = ?", (session_id,))
    session_info = cursor.fetchone()
    # 只有 update_timestamp 明确为 True 时才更新时间戳
    if str(update_timestamp).lower() == 'true' and session_info:
        # 使用标准ISO格式的时间戳，确保SQLite可以正确解析排序
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "UPDATE chat_sessions SET timestamp = ? WHERE id = ?",
            (timestamp, session_id)
        )
        conn.commit()
        session_info = dict(session_info)
        session_info["timestamp"] = timestamp
    result = {
        "messages": [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
    }
    if session_info:
        result["summary"] = session_info["summary"]
        result["timestamp"] = session_info["timestamp"]
    conn.close()
    return result

# 联网搜索函数
async def web_search(query, num_results=5):
    """
    使用SerpAPI进行联网搜索，获取实时信息
    
    参数:
        query: 搜索查询文本
        num_results: 返回的结果数量
        
    返回:
        包含搜索结果的字符串
    """
    try:
        # 如果没有API密钥，则返回错误消息
        if not SERPAPI_API_KEY:
            return "搜索功能暂时不可用，请联系管理员配置SERPAPI_API_KEY。"
        
        # 构建基本的请求参数
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": str(num_results),
            "hl": "zh-cn",     # 语言设置为中文
            "gl": "cn",        # 地理位置设置为中国
            "google_domain": "google.com"
        }
        
        print(f"SerpAPI 请求参数: {params}")
        
        # 发送请求到SerpAPI
        response = requests.get(
            "https://serpapi.com/search", 
            params=params,
            timeout=15  # 设置15秒超时
        )
        
        # 检查响应状态
        if response.status_code != 200:
            print(f"SerpAPI 请求失败，状态码：{response.status_code}，响应内容：{response.text}")
            return f"搜索请求失败，状态码：{response.status_code}"
        
        print(f"SerpAPI 请求成功，状态码：{response.status_code}")
        data = response.json()
        
        # 提取有用的搜索结果
        search_results = []
        
        # 从顶部摘要提取信息（如果有）
        if "answer_box" in data:
            answer = data["answer_box"].get("answer") or data["answer_box"].get("snippet")
            if answer:
                search_results.append(f"摘要：{html.unescape(answer)}")
        
        # 从知识图谱提取信息（如果有）
        if "knowledge_graph" in data and "description" in data["knowledge_graph"]:
            search_results.append(f"知识图谱：{html.unescape(data['knowledge_graph']['description'])}")
        
        # 提取有机搜索结果
        if "organic_results" in data:
            for result in data["organic_results"][:num_results]:
                title = html.unescape(result.get("title", "无标题"))
                snippet = html.unescape(result.get("snippet", "无摘要"))
                link = result.get("link", "")
                search_results.append(f"标题：{title}\n摘要：{snippet}\n链接：{link}\n")
        
        # 如果没有搜索结果
        if not search_results:
            return "未找到相关搜索结果。"
        
        # 组合搜索结果
        final_result = "\n\n".join(search_results)
        print(f"搜索结果长度: {len(final_result)}")
        return final_result
    
    except Exception as e:
        print(f"联网搜索出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"联网搜索过程中出现错误: {str(e)}"

@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """
    流式对话接口，支持历史上下文和知识库检索
    前端需POST: {"query": "...", "session_id": "...", "use_web_search": true/false}
    """
    try:
        data = await request.json()
        query = data.get("query", "")
        session_id = data.get("session_id") or str(uuid4())
        use_web_search = data.get("use_web_search", False)  # 是否使用联网搜索
        is_new_session = False
        is_session_created = False  # 标记是否新创建了会话
        
        # 检查会话是否已存在
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,))
        existing_session = cursor.fetchone()
        if not existing_session:
            # 新会话，需要立即创建并保存到数据库
            is_new_session = True
            is_session_created = True
            # 使用标准ISO格式的时间戳，确保SQLite可以正确解析排序
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 初始创建时使用"新会话"作为标题
            cursor.execute(
                "INSERT INTO chat_sessions (id, summary, custom_summary, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, "新会话", 0, timestamp)
            )
            conn.commit()
            print(f"新会话已创建: ID={session_id}, 摘要=新会话")
            
        # 从数据库获取会话历史
        cursor.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id", (session_id,))
        history_rows = cursor.fetchall()
        conn.close()
        history = [dict(row) for row in history_rows] if history_rows else []
        
        # 检索知识库文档
        retrieved_docs, retrieved_chunks = retrieve_docs(query, index)
        
        # 构建上下文，包含历史消息和知识库内容
        context = ""
        for msg in history:
            if msg["role"] == "user":
                context += f"用户：{msg['content']}\n"
            else:
                context += f"助手：{msg['content']}\n"
        context += f"用户：{query}\n"
        
        # 如果启用了联网搜索，执行搜索
        web_search_results = ""
        if use_web_search:
            print(f"启用联网搜索，关键词: {query}")
            web_search_results = await web_search(query)
            context += f"\n联网搜索结果:\n{web_search_results}\n"
        
        # 拼接知识库内容
        kb_context = "\n知识库相关内容：\n"
        for doc_id, chunk in retrieved_chunks:
            kb_context += f"[文档{doc_id}] {chunk}\n"
        context += kb_context
        context += "助手："
        
        # 保存用户消息到数据库(会更新时间戳)
        save_message(session_id, "user", query, is_new_message=True)
        
        # 如果是第一条消息，使用它作为摘要更新会话标题
        if is_new_session or len(history) == 0:
            conn = get_db_connection()
            cursor = conn.cursor()
            summary = query[:20] + ("..." if len(query) > 20 else "")
            cursor.execute(
                "UPDATE chat_sessions SET summary = ? WHERE id = ?",
                (summary, session_id)
            )
            conn.commit()
            conn.close()
            print(f"更新会话摘要: ID={session_id}, 摘要={summary}")
        
        return StreamingResponse(
            generate_chat_stream(context, session_id, is_session_created, is_new_session, len(history) == 0, use_web_search),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked"
            }
        )
    except Exception as e:
        print(f"聊天流错误: {str(e)}")
        def error_response():
            yield f"data: {json.dumps({'error': '处理请求失败，请稍后再试'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_response(), media_type="text/event-stream")

async def generate_chat_stream(context, session_id, is_session_created, is_new_session, is_first_message, web_search_used=False):
    """流式生成聊天响应"""
    try:
        system_message = "你是一个专业的问答助手。请仅基于提供的上下文信息和知识库内容回答问题，不要添加任何未在上下文中提及的信息。"
        if web_search_used:
            system_message += "你现在拥有搜索互联网的能力，可以获取实时信息。如果上下文中含有联网搜索结果，可以使用这些信息来辅助回答。"
        
        # 流式大模型调用
        stream = client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": context}
            ],
            stream=True
        )
        
        assistant_reply = ""
        # 如果是新创建的会话，先发送一个通知让前端知道需要刷新会话列表
        if is_session_created:
            yield f"data: {json.dumps({'session_created': True, 'session_id': session_id})}\n\n"
            await asyncio.sleep(0.05)  # 适当延迟，确保前端有时间处理
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                assistant_reply += content
                # 直接发送每个内容片段
                yield f"data: {json.dumps({'content': content})}\n\n"
                
                # 根据内容长度动态调整延迟，提供更好的打字机效果
                if len(content) > 5:
                    # 内容较长时使用较长延迟
                    await asyncio.sleep(0.03)
                else:
                    # 内容短时使用较短延迟
                    await asyncio.sleep(0.01)
                
            if chunk.choices[0].finish_reason is not None:
                # 发送结束标记
                yield f"data: [DONE]\n\n"
                # 保存助手回复到数据库(会更新时间戳)
                save_message(session_id, "assistant", assistant_reply, is_new_message=True)
                break
    except Exception as e:
        print(f"流式响应处理错误: {str(e)}")
        yield f"data: {json.dumps({'error': '响应处理失败'})}\n\n"
        yield f"data: [DONE]\n\n"
        # 确保仍然保存助手回复，即使不完整
        if "assistant_reply" in locals() and assistant_reply:
            save_message(session_id, "assistant", assistant_reply, is_new_message=True)

@app.post("/api/chat/save")
async def save_chat_session(request: Request):
    """保存会话消息"""
    try:
        # 解析请求数据
        data = await request.json()
        session_id = data.get("session_id", "")
        messages = data.get("messages", [])
        summary = data.get("summary", "新会话")
        custom_timestamp = data.get("timestamp")  # 前端可能提供时间戳
        update_timestamp = data.get("update_timestamp", True)  # 是否更新时间戳
        
        if not session_id:
            session_id = str(uuid4())
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 使用前端提供的时间戳或生成新的，确保格式标准化
        if custom_timestamp and update_timestamp:
            # 尝试规范化前端提供的时间戳
            try:
                # 如果包含斜杠，尝试转换为标准格式
                if '/' in custom_timestamp:
                    date_part, time_part = custom_timestamp.split(' ', 1)
                    year, month, day = date_part.split('/')
                    # 确保年月日格式统一
                    year = year.zfill(4)
                    month = month.zfill(2)
                    day = day.zfill(2)
                    timestamp = f"{year}-{month}-{day} {time_part}"
                else:
                    # 已经是标准格式
                    timestamp = custom_timestamp
            except Exception:
                # 转换失败，使用当前时间
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elif update_timestamp:
            # 生成新的标准格式时间戳
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            # 不更新时间戳，获取现有时间戳
            cursor.execute("SELECT timestamp FROM chat_sessions WHERE id = ?", (session_id,))
            result = cursor.fetchone()
            timestamp = result["timestamp"] if result else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 检查会话是否已存在
        cursor.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,))
        session = cursor.fetchone()
        
        if not session:
            # 创建新会话
            cursor.execute(
                "INSERT INTO chat_sessions (id, summary, custom_summary, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, summary, 1, timestamp)
            )
            print(f"创建新会话，ID: {session_id}, 摘要: {summary}")
        else:
            # 更新现有会话
            cursor.execute(
                "UPDATE chat_sessions SET summary = ?, timestamp = ? WHERE id = ?",
                (summary, timestamp, session_id)
            )
            print(f"更新现有会话，ID: {session_id}, 摘要: {summary}, 时间戳: {timestamp}")
        
        # 删除现有消息
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        
        # 插入新消息
        for message in messages:
            cursor.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, message["role"], message["content"], timestamp)
            )
        
        conn.commit()
        conn.close()
        
        # 返回更新后的数据，确保包含summary字段
        return {
            "message": "保存成功", 
            "session_id": session_id,
            "summary": summary,
            "timestamp": timestamp
        }
    except Exception as e:
        print(f"保存会话失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"保存会话失败: {str(e)}"}

@app.delete("/api/chat/session/{session_id}")
def delete_chat_session(session_id: str):
    """删除会话及其所有相关消息"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 检查会话是否存在
        cursor.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,))
        session = cursor.fetchone()
        
        if not session:
            conn.close()
            return {"error": "会话不存在"}
        
        # 开始事务，确保操作的原子性
        cursor.execute("BEGIN TRANSACTION")
        
        try:
            # 首先删除会话相关的所有消息
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            deleted_msg_count = cursor.rowcount
            
            # 然后删除会话本身
            cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            
            # 提交事务
            conn.commit()
            
            print(f"已删除会话ID: {session_id}，相关消息数: {deleted_msg_count}")
            conn.close()
            
            return {
                "message": "删除成功",
                "deleted_messages": deleted_msg_count
            }
        except Exception as e:
            # 发生错误时回滚事务
            cursor.execute("ROLLBACK")
            conn.close()
            raise e
            
    except Exception as e:
        print(f"删除会话失败: {str(e)}")
        return {"error": f"删除会话失败: {str(e)}"}

# ========== 新增：重建分块和向量数据库函数 ==========
def rebuild_vector_index():
    """
    重新加载所有文档，分块，生成嵌入，重建FAISS索引，并保存到数据库
    """
    print("[索引重建] 正在重建向量索引...")
    documents, sources, errors = load_documents_from_directory("./docs")
    print(f"[索引重建] 加载文档数: {len(documents)}，错误: {errors}")
    document_to_chunks = {}
    chunks_to_document = {}
    all_chunks = []
    for doc_id, doc in enumerate(documents):
        chunks = chunk_document(doc)
        document_to_chunks[doc_id] = []
        for chunk in chunks:
            chunk_id = len(all_chunks)
            all_chunks.append(chunk)
            document_to_chunks[doc_id].append(chunk_id)
            chunks_to_document[chunk_id] = doc_id
    print(f"[索引重建] 总分块数: {len(all_chunks)}")
    if all_chunks:
        chunk_embeddings = get_embeddings(all_chunks)
        print(f"[索引重建] 嵌入生成完成，维度: {chunk_embeddings.shape}")
        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(chunk_embeddings)
        mapping_data = {
            'doc_to_chunks': document_to_chunks,
            'chunks_to_doc': chunks_to_document,
            'all_chunks': all_chunks
        }
        save_vector_index_to_db(index, mapping_data)
        print("[索引重建] 向量索引重建完成")
    else:
        print("[索引重建] 没有可用文档，未重建索引")

@app.get("/api/chat/stream")
async def chat_stream_get(
    query: str,
    session_id: str = None,
    use_web_search: bool = Query(False)  # 添加联网搜索参数
):
    """
    通过GET方式提供的流式对话接口，支持EventSource API使用
    前端需GET请求: /api/chat/stream?query=问题内容&session_id=会话ID&use_web_search=true/false
    """
    if not session_id:
        session_id = str(uuid4())
    
    # 采用与POST接口相同的处理逻辑
    is_new_session = False
    is_session_created = False

    # 检查会话是否已存在
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,))
    existing_session = cursor.fetchone()
    
    if not existing_session:
        # 新会话，需要立即创建并保存到数据库
        is_new_session = True
        is_session_created = True
        # 使用标准ISO格式的时间戳，确保SQLite可以正确解析排序
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 初始创建时使用"新会话"作为标题
        cursor.execute(
            "INSERT INTO chat_sessions (id, summary, custom_summary, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, "新会话", 0, timestamp)
        )
        conn.commit()
        print(f"新会话已创建: ID={session_id}, 摘要=新会话")

    # 从数据库获取会话历史
    cursor.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id", (session_id,))
    history_rows = cursor.fetchall()
    conn.close()
    history = [dict(row) for row in history_rows] if history_rows else []
    
    # 检索知识库文档
    retrieved_docs, retrieved_chunks = retrieve_docs(query, index)
    
    # 构建上下文，包含历史消息和知识库内容
    context = ""
    for msg in history:
        if msg["role"] == "user":
            context += f"用户：{msg['content']}\n"
        else:
            context += f"助手：{msg['content']}\n"
    context += f"用户：{query}\n"
    
    # 如果启用了联网搜索，执行搜索
    web_search_results = ""
    if use_web_search:
        print(f"启用联网搜索，关键词: {query}")
        web_search_results = await web_search(query)
        context += f"\n联网搜索结果:\n{web_search_results}\n"
    
    # 拼接知识库内容
    kb_context = "\n知识库相关内容：\n"
    for doc_id, chunk in retrieved_chunks:
        kb_context += f"[文档{doc_id}] {chunk}\n"
    context += kb_context
    context += "助手："
    
    # 保存用户消息到数据库(会更新时间戳)
    save_message(session_id, "user", query, is_new_message=True)
    
    # 如果是第一条消息，使用它作为摘要更新会话标题
    if is_new_session or len(history) == 0:
        conn = get_db_connection()
        cursor = conn.cursor()
        summary = query[:20] + ("..." if len(query) > 20 else "")
        cursor.execute(
            "UPDATE chat_sessions SET summary = ? WHERE id = ?",
            (summary, session_id)
        )
        conn.commit()
        conn.close()
        print(f"更新会话摘要: ID={session_id}, 摘要={summary}")
    
    return StreamingResponse(
        generate_chat_stream(context, session_id, is_session_created, is_new_session, len(history) == 0, use_web_search),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked"
        }
    )

# 启动main函数
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 