from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from tools import time_tool,save_tool,read_doc_tool,run_code_tool,read_python_file,save_python_file

messages_history=[]

# 加载 .env 文件中的环境变量
# load_dotenv() 会查找项目根目录下的 .env 文件并加载其中的变量到环境变量中
load_dotenv()

# 从环境变量中读取配置
# os.getenv(key, default) - 如果环境变量不存在，使用默认值
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

# 使用环境变量创建 LLM 实例
# 优点：
# 1. API Key 不会硬编码在代码中，更安全
# 2. 可以轻松切换不同的 API 配置
# 3. 不同环境（开发/测试/生产）可以使用不同的配置
llm=ChatOpenAI(
    model=DEEPSEEK_MODEL,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    base_url=DEEPSEEK_BASE_URL,
    api_key=DEEPSEEK_API_KEY
)

print("----创建llm完成----")
print("LLM："+DEEPSEEK_MODEL)

tools=[time_tool,save_tool,read_doc_tool,run_code_tool,read_python_file,save_python_file]

print("----工具创建完成----")
for x in tools:
    print("工具："+x.name)

system_prompt="""
你是一个智能编程助手，具备以下核心能力：
1. 读取、分析和修改 Python 文件
2. 执行 Python 代码并解释结果
3. 读取和分析 Word 文档
4. 搜索网络信息
5. 保存数据到文件

【核心功能】

## 1. Python 文件管理功能（最重要）
当用户要求读取、分析或修改 Python 文件时：

**读取文件：**
- 使用 read_python_file 工具读取 .py 文件
- 文件内容会带行号显示，方便定位
- 仔细阅读代码，理解其功能和结构

**修改文件：**
- 根据用户需求分析需要修改的部分
- 编写修改后的完整代码
- 使用 save_python_file 工具保存修改
- 参数：code（完整代码）, file_path（文件路径）

**工作流程 - 修改代码：**
第1步：读取原文件
   - 调用 read_python_file(file_path)
   - 仔细分析代码结构和逻辑

第2步：理解用户需求
   - 用户想要添加什么功能？
   - 需要修复什么问题？
   - 需要优化哪些部分？

第3步：修改代码
   - 在原有代码基础上进行修改
   - 保持代码风格和结构
   - 添加必要的注释
   - 确保代码逻辑正确

第4步：保存修改
   - 调用 save_python_file(code, file_path)
   - 传入完整的修改后代码
   - 确认保存成功

第5步：测试验证（可选）
   - 使用 run_python_code 运行修改后的文件
   - 验证功能是否正常

## 2. Python 代码执行功能
当用户要求运行、测试或解释 Python 代码时：
- 理解用户的代码需求
- 编写或修改 Python 代码
- 使用 run_python_code 工具执行代码
- 解释代码的执行结果和输出
- 如有错误，分析原因并提供修复建议

## 3. 文档总结功能
当用户要求总结 Word 文档时：
- 使用 read_document 工具读取文档内容
- 分析文档结构和关键信息
- 生成结构化总结报告
- 使用 save_text_to_file 保存结果

【可用工具】
1. read_python_file(file_path) - 读取Python文件（带行号）
2. save_python_file(code, file_path) - 保存/修改Python文件
3. run_python_code(code) - 执行Python代码或运行.py文件
4. read_document(file_path) - 读取Word文档内容
5. save_text_to_file(data, filename) - 保存文本到文件
6. get_time() - 获取当前时间
7. search(query) - 搜索网络信息
8. wikipedia(query) - 查询维基百科

【重要提醒】
- 修改代码前，先读取原文件了解现有代码
- 保持代码的完整性和可运行性
- 修改后说明做了哪些改动
- 执行代码前，向用户说明将要运行的代码
- 代码执行后，详细解释输出结果
- 如果代码出错，分析错误原因并提供修复方案
- 对于复杂任务，分步骤执行

【示例场景】
1. "帮我读取 main.py 并添加一个日志功能"
2. "查看 test.py 的代码，然后修复其中的bug"
3. "在 simple_test.py 中添加一个排序函数"
4. "运行 test_code_execution.py 并解释输出"

现在，请等待用户输入，然后根据上述流程执行任务。
"""



print("----提示词创建完成----")


agent=create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

print("----agent创建完成----")

def chat_with_agent(user_input: str):
    global messages_history
    print("思考中......")
    all_messages=messages_history.copy()
    all_messages.append(user_input)
    print(all_messages)

    response=agent.invoke({"messages":all_messages})
    try:
        if 'messages' in response and len(response['messages'])>0 :
            ai_response=response['messages'][-1].content


            messages_history.append(("human", user_input))
            messages_history.append(("ai", ai_response))

            # 打印中间步骤（调试用）
            # print("\n--- Agent 执行过程 ---")
            # for msg in response['messages']:
            #     if hasattr(msg, 'type'):
            #         print(f"[{msg.type}]: {str(msg.content)[:200]}...")
            # print("--- 执行过程结束 ---\n")

        else:
            ai_response=str(response)

    except Exception as e:
        ai_response = str(e)
        print("出错")
        print("出错内容为："+ai_response)

    return ai_response

print("----聊天模块创建完成----")


while True:
    user_input = input("你: ")
    if user_input == "q":
        break
    ai_response = chat_with_agent(user_input)
    print("AI: " + ai_response)

