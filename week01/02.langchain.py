import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool
import datetime

# 加载环境变量
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_API_BASE')

# ========== 1. 初始化 LLM（大语言模型） ==========
# LangChain 特点：统一的 LLM 接口，支持多种模型提供商
llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model="qwen3-max",
    temperature=0.7  # LangChain 特点：统一的参数配置
)

# ========== 2. LLMChain：Prompt → LLM → 输出链的基本流程封装 ==========
def demo_llm_chain():
    """
    演示 LLMChain：支持变量注入与模板复用的核心组件
    LangChain 特点：模板化提示词管理，支持变量替换
    """
    print("=" * 50)
    print("🔗 LLMChain 演示：Prompt → LLM → 输出链")
    print("=" * 50)
    
    # 创建提示词模板 - LangChain 特点：模板复用
    prompt_template = PromptTemplate(
        input_variables=["topic", "style"],
        template="""
        请以{style}的风格，写一段关于{topic}的介绍。
        要求：简洁明了，不超过100字。
        """
    )
    
    # LangChain 0.3 推荐使用 LCEL (LangChain Expression Language)
    # 这是新的链式组合方式：prompt | llm
    chain = prompt_template | llm
    
    # 执行链 - 变量注入
    result = chain.invoke({"topic": "人工智能", "style": "科普"})
    print(f"📝 LLMChain 输出：\n{result.content}\n")
    
    return result.content

# ========== 3. Tools：工具系统 ==========
def get_current_time(query: str) -> str:
    """获取当前时间的工具函数"""
    return f"当前时间是：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def calculate_simple(expression: str) -> str:
    """简单计算器工具"""
    try:
        # 安全的数学表达式计算
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"计算结果：{expression} = {result}"
        else:
            return "错误：包含不允许的字符"
    except Exception as e:
        return f"计算错误：{str(e)}"

# LangChain 特点：统一的工具接口定义
tools = [
    Tool(
        name="get_time",
        func=get_current_time,
        description="获取当前的日期和时间信息"
    ),
    Tool(
        name="calculator",
        func=calculate_simple,
        description="执行简单的数学计算，如加减乘除运算"
    )
]

def demo_tools():
    """演示 Tools 工具系统"""
    print("=" * 50)
    print("🛠️ Tools 演示：工具系统")
    print("=" * 50)
    
    for tool in tools:
        print(f"工具名称：{tool.name}")
        print(f"工具描述：{tool.description}")
        
        # 测试工具
        if tool.name == "get_time":
            result = tool.run("现在几点了？")
        else:
            result = tool.run("10 + 5 * 2")
        
        print(f"工具输出：{result}\n")

# ========== 4. 简化版 Agents：手动工具选择演示 ==========
def demo_simple_agents():
    """
    演示简化版 Agents：手动工具选择和执行
    LangChain 特点：工具集成和智能选择（这里用简化版演示概念）
    """
    print("=" * 50)
    print("🤖 简化版 Agents 演示：工具选择与执行")
    print("=" * 50)
    
    # 创建工具选择提示词
    tool_selection_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能助手，可以使用以下工具：
        1. get_time - 获取当前时间
        2. calculator - 执行数学计算
        
        请分析用户问题，选择合适的工具并说明原因。
        只回答工具名称和原因，格式：工具名称|原因"""),
        ("human", "{question}")
    ])
    
    tool_chain = tool_selection_prompt | llm
    
    test_questions = [
        "现在几点了？",
        "帮我计算 15 * 8 + 20",
        "今天是什么日期？"
    ]
    
    for question in test_questions:
        print(f"👤 用户问题：{question}")
        
        # 1. 工具选择
        selection_result = tool_chain.invoke({"question": question})
        print(f"🧠 工具选择：{selection_result.content}")
        
        # 2. 执行工具（简化版手动执行）
        if "get_time" in selection_result.content.lower():
            result = get_current_time(question)
        elif "calculator" in selection_result.content.lower():
            # 提取数学表达式（简化处理）
            if "15 * 8 + 20" in question:
                result = calculate_simple("15 * 8 + 20")
            else:
                result = "需要具体的数学表达式"
        else:
            result = "未找到合适的工具"
        
        print(f"🛠️ 工具执行结果：{result}\n")

# ========== 5. Memory：记忆系统 ==========
def demo_memory():
    """
    演示 Memory：对话记忆管理
    LangChain 特点：自动管理对话历史
    """
    print("=" * 50)
    print("🧠 Memory 演示：记忆系统")
    print("=" * 50)
    
    # 使用简化的记忆管理方式
    conversation_history = []
    
    # 创建带记忆的对话提示词
    memory_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手，能够记住之前的对话内容。以下是对话历史：{history}"),
        ("human", "{input}")
    ])
    
    memory_chain = memory_prompt | llm
    
    # 模拟多轮对话
    conversations = [
        "我叫张三，今年25岁",
        "我喜欢编程和阅读",
        "你还记得我的名字吗？",
        "我的爱好是什么？"
    ]
    
    for i, user_input in enumerate(conversations, 1):
        print(f"👤 第{i}轮对话：{user_input}")
        
        # 构建历史记录字符串
        history_str = "\n".join([f"用户: {h['user']}\n助手: {h['assistant']}" for h in conversation_history])
        
        # 获取回复
        response = memory_chain.invoke({
            "history": history_str,
            "input": user_input
        })
        
        print(f"🤖 助手回复：{response.content}\n")
        
        # 更新对话历史
        conversation_history.append({
            "user": user_input,
            "assistant": response.content
        })
        
        # 显示当前记忆内容
        print(f"💭 当前记忆：{len(conversation_history)} 轮对话")
        print("-" * 30)

# ========== 6. LCEL 演示：LangChain Expression Language ==========
def demo_lcel():
    """
    演示 LCEL：LangChain 0.3 的新特性
    LangChain 特点：更简洁的链式组合语法
    """
    print("=" * 50)
    print("🔗 LCEL 演示：LangChain Expression Language")
    print("=" * 50)
    
    # LCEL 语法：使用 | 操作符组合组件
    prompt = PromptTemplate.from_template("请用{language}语言解释什么是{concept}")
    
    # 创建链：prompt | llm
    chain = prompt | llm
    
    # 执行链
    result = chain.invoke({
        "language": "简单易懂的中文",
        "concept": "区块链"
    })
    
    print(f"📝 LCEL 链式调用结果：\n{result.content}\n")
    
    # 演示更复杂的链组合
    from langchain_core.output_parsers import StrOutputParser
    
    # 创建输出解析器
    output_parser = StrOutputParser()
    
    # 更复杂的链：prompt | llm | output_parser
    complex_chain = prompt | llm | output_parser
    
    result2 = complex_chain.invoke({
        "language": "技术术语",
        "concept": "机器学习"
    })
    
    print(f"📝 复杂 LCEL 链结果：\n{result2}\n")

# ========== 7. 综合演示：LangChain 特点总结 ==========
def demo_langchain_features():
    """展示 LangChain 的核心特点"""
    print("=" * 60)
    print("🌟 LangChain 核心特点总结")
    print("=" * 60)
    
    features = [
        "🔗 链式组合：使用 LCEL (|) 将多个组件串联",
        "📝 模板管理：统一的提示词模板系统",
        "🛠️ 工具集成：标准化的工具接口",
        "🤖 智能代理：自动选择和使用工具（需模型支持）",
        "🧠 记忆管理：灵活的对话历史管理",
        "🔄 流程编排：灵活的工作流定义",
        "📊 可观测性：详细的执行日志",
        "🔌 模块化：组件可插拔设计",
        "⚡ LCEL：简洁的表达式语言",
        "🎯 类型安全：完整的类型提示支持"
    ]
    
    for feature in features:
        print(feature)
    
    print("\n" + "=" * 60)

def main():
    """主函数：依次演示各个核心组件"""
    print("🚀 LangChain 0.3 核心组件实战演示")
    print("基于 OpenAI API 的完整示例（兼容版本）\n")
    
    try:
        # 1. LLMChain 演示（使用 LCEL）
        demo_llm_chain()
        
        # 2. Tools 演示
        demo_tools()
        
        # 3. 简化版 Agents 演示
        demo_simple_agents()
        
        # 4. Memory 演示
        demo_memory()
        
        # 5. LCEL 演示
        demo_lcel()
        
        # 6. 特点总结
        demo_langchain_features()
        
        print("✅ 所有演示完成！")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误：{str(e)}")
        print("请检查 API 密钥和网络连接")

if __name__ == "__main__":
    main()