# 导入依赖
# pip install llama-index llama-index-embeddings-dashscope llama-index-llms-openai-like
import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.embeddings.dashscope import DashScopeEmbedding,DashScopeTextEmbeddingModels
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike

# 这两行代码是用于消除 WARNING 警告信息，避免干扰阅读学习，生产环境中建议根据需要来设置日志级别
import logging
logging.basicConfig(level=logging.ERROR)
load_dotenv()

print("正在解析文件...")

# 获取当前脚本的绝对路径
script_path = Path(__file__).resolve()
# 获取脚本所在的目录
script_dir = script_path.parent
# 构建docs目录的路径（与脚本同级）
docs_path = script_dir / "docs"

print(docs_path)

# LlamaIndex提供了SimpleDirectoryReader方法，可以直接将指定文件夹中的文件加载为document对象，对应着解析过程
documents = SimpleDirectoryReader(str(docs_path)).load_data()

print("正在创建索引...")
# from_documents方法包含切片与建立索引步骤
index = VectorStoreIndex.from_documents(
    documents,
    # 指定embedding 模型
    embed_model=DashScopeEmbedding(
        # 你也可以使用阿里云提供的其它embedding模型：https://help.aliyun.com/zh/model-studio/getting-started/models#3383780daf8hw
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2
    ))
print("正在创建提问引擎...")
query_engine = index.as_query_engine(
    # 设置为流式输出
    streaming=True,
    # 此处使用qwen-plus模型，你也可以使用阿里云提供的其它qwen的文本生成模型：https://help.aliyun.com/zh/model-studio/getting-started/models#9f8890ce29g5u
    llm=OpenAILike(
        model="qwen-plus",
        api_base=os.getenv("DASHSCOPE_API_BASE"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        is_chat_model=True
        ))
print("正在生成回复...")
streaming_response = query_engine.query('我们公司项目管理应该用什么工具')
print("回答是：")
# 采用流式输出
streaming_response.print_response_stream()