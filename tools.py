# ============================================================================
# 文件名: tools_file.py
# 功能: 定义研究助手 Agent 可用的工具集
# 说明: Agent 通过这些工具来执行实际操作（搜索、查询、保存数据等）
# ============================================================================

# ============================================================================
# 第1部分: 导入依赖库
# ============================================================================
from langchain_core.tools import Tool, tool      # LangChain 工具类：将普通 Python 函数包装成 Agent 可调用的工具
from datetime import datetime              # Python 标准库：处理日期和时间（用于生成时间戳）
import urllib.request                      # Python 标准库：HTTP 请求（预留使用，当前未用到）
import json                                # Python 标准库：处理 JSON 数据（预留使用，当前未用到）
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

import os
import subprocess
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# 数据分析和可视化库
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import seaborn as sns

# 向量数据库和嵌入模型
from langchain_chroma import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import chromadb



# ============================================================================
# 第2部分: 工具函数定义
# ============================================================================

# ----------------------------------------------------------------------------
# 工具1: save_to_txt - 文件保存工具
# 功能: 将研究数据保存到本地文本文件，并自动添加时间戳
# 使用场景: 保存研究成果、日志记录、数据持久化
# ----------------------------------------------------------------------------
def save_to_txt(data: str, filename: str = None):
    """
    将研究数据保存到文本文件（自动使用时间戳生成唯一文件名）
    
    Args:
        data: 要保存的研究数据内容（字符串格式）
        filename: 文件名（可选，如果不提供则自动生成带时间戳的文件名）
                  格式: research_output_2026-04-10_15-30-25.txt
    
    Returns:
        str: 保存成功的确认信息

    """
    # 如果没有提供文件名，则根据当前时间自动生成唯一文件名
    if filename is None:
        # 生成格式：research_output_年-月-日_时-分-秒.txt
        # 例如：research_output_2026-04-10_15-30-25.txt
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"生成文件_{timestamp}.txt"
        print("自动生成文件名："+filename)
    
    # 获取当前系统时间，用于文件内容中的时间戳
    timestamp_content = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 使用 f-string 拼接格式化的文本
    # 结构: 标题 → 时间戳 → 空行 → 数据内容 → 空行
    # 例如:
    # --- Research Output ---
    # Timestamp: 2026-04-10 15:30:25
    #
    # Python 是一种高级编程语言...
    #
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp_content}\n\n{data}\n\n"

    # 以追加模式打开文件并写入数据
    # open(filename, "a", encoding="utf-8"):
    #   - "a" 模式: Append（追加），不会覆盖已有内容，新内容添加到文件末尾
    #   - encoding="utf-8": 支持中文和其他 Unicode 字符，避免乱码
    # with 语句: 上下文管理器，自动在代码块结束后关闭文件（即使发生异常）
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    # 返回确认信息给 Agent，让 Agent 知道文件保存成功
    return f"Data successfully saved to {filename}"


# ----------------------------------------------------------------------------
# 工具2: web_search - 网页搜索工具
# 功能: 使用 DuckDuckGo 搜索引擎查询互联网信息
# 特点: 免费、无需 API Key、隐私保护
# 使用场景: 获取最新的网络信息、新闻、教程等
# ----------------------------------------------------------------------------
def web_search(query: str) -> str:
    """
    使用 DuckDuckGo 搜索网页信息
    
    Args:
        query: 搜索关键词（例如："Python programming"、"人工智能最新进展"）
    
    Returns:
        str: 搜索结果文本，包含标题和摘要（最多3条）
    
    返回格式示例:
        "Python (programming language): Python is a high-level, general-purpose...
        
        Guido van Rossum: Guido van Rossum is a Dutch programmer...
        
        Python Software Foundation: The Python Software Foundation is..."
    
    异常处理:
        如果搜索失败（网络错误、API 限制等），返回错误信息而不是抛出异常
    """
    print("正在搜索："+query)
    # 使用 try-except 捕获所有可能的异常
    # 网络请求可能失败：断网、DNS 解析失败、API 限流等
    try:
        # 延迟导入：只在函数执行时才导入模块
        # 优点：
        #   1. 避免启动时就加载所有依赖
        #   2. 如果模块未安装，不影响其他功能
        #   3. 加快模块初始化速度
        from ddgs import DDGS  # DuckDuckGo 搜索的 Python 客户端库
        
        # 使用 with 语句创建搜索客户端
        # DDGS() 初始化搜索引擎客户端
        # with 语句确保使用完毕后自动释放资源（如关闭网络连接）
        with DDGS() as ddgs:
            # 执行文本搜索
            # query: 用户输入的搜索关键词
            # max_results=3: 限制返回结果数量（避免信息过载，节省 token）
            results = ddgs.text(query, max_results=3)
            
            # 检查是否有搜索结果
            if results:
                # 使用列表推导式格式化搜索结果
                # 原始结果格式: [{"title": "...", "body": "...", "href": "..."}, ...]
                # 目标格式: "标题1: 摘要1\n\n标题2: 摘要2\n\n标题3: 摘要3"
                #
                # 列表推导式详解:
                # [f"{r['title']}: {r['body']}" for r in results]
                #   ↓ 等价于 ↓
                # formatted_list = []
                # for r in results:  # 遍历每个搜索结果
                #     formatted_item = f"{r['title']}: {r['body']}"  # 格式化
                #     formatted_list.append(formatted_item)
                #
                # "\n\n".join(...): 用双换行符连接所有结果
                return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
            
            # 如果没有搜索结果，返回提示信息
            return "No results found"
    
    except Exception as e:
        # 捕获所有异常并返回错误信息
        # str(e): 将异常对象转换为字符串（错误描述）
        # 这样 Agent 能看到错误信息并决定下一步操作
        return f"Search failed: {str(e)}"


# ----------------------------------------------------------------------------
# 工具3: wiki_search - 维基百科搜索工具
# 功能: 从维基百科获取主题的权威摘要信息
# 特点: 内容权威、结构化、适合查询概念和背景知识
# 使用场景: 查询名词解释、历史背景、科学概念等
# ----------------------------------------------------------------------------
def wiki_search(query: str) -> str:
    """
    从维基百科查询信息
    
    Args:
        query: 查询主题（例如："Machine learning"、"Python programming"）
    
    Returns:
        str: 维基百科词条摘要（最多3句话）
    
    返回示例:
        "Python is a high-level, general-purpose programming language. Its design 
        philosophy emphasizes code readability with the use of significant 
        indentation. Python is dynamically typed and garbage-collected."
    
    异常处理:
        如果词条不存在、网络错误或语言不支持，返回错误信息
    """
    # 使用 try-except 处理可能的异常
    print("正在查询维基百科...")
    try:
        # 延迟导入维基百科 API 客户端
        import wikipedia  # Python 维基百科库
        
        # 设置查询语言为英语
        # wikipedia.set_lang("en"):
        #   - "en": 英语维基百科（内容最全面，1000万+词条）
        #   - 可选值: "zh"（中文）、"ja"（日语）、"fr"（法语）等
        # 为什么选择英语？
        #   1. 技术类词条英文最全面、最准确
        #   2. 大多数学术概念的标准命名是英文
        #   3. 中文维基百科可能缺少某些专业术语
        wikipedia.set_lang("zh")
        
        # 获取词条摘要
        # wikipedia.summary(query, sentences=3):
        #   - query: 查询的词条名称
        #   - sentences=3: 限制返回3句话（避免返回整篇长文，节省 token）
        #   - 返回的是词条第一段的精简版（摘要部分）
        summary = wikipedia.summary(query, sentences=3)
        
        # 返回摘要文本给 Agent
        return summary
    
    except Exception as e:
        # 常见异常：
        #   - DisambiguationError: 词条名不明确，有多个匹配（如 "Python" 可以是编程语言或蟒蛇）
        #   - PageError: 词条不存在
        #   - HTTPError: 网络请求失败
        return f"Wikipedia search failed: {str(e)}"

# ----------------------------------------------------------------------------
# 工具4: exit_agent - 退出程序工具
# 功能: 当用户说再见时，优雅地退出程序
# 注意: 必须接受一个参数（LangChain Tool 调用时会传入参数）
# ----------------------------------------------------------------------------
def exit_agent(_input: str = "") -> str:
    """
    退出 Agent 程序
    
    Args:
        _input: 占位符参数（LangChain Tool 调用时必须传入，但这里不使用）
    
    Returns:
        str: 退出确认信息
    """
    print("\n👋 Goodbye! 感谢使用，再见！")
    exit(0)
    # 返回消息让 Agent 知道已退出（实际上程序会立即终止）


# ----------------------------------------------------------------------------
# 工具5: get_datatime - 获取当前时间工具
# 功能: 返回当前日期和时间
# 注意: 必须接受一个参数（LangChain Tool 调用时会传入参数）
# ----------------------------------------------------------------------------
def get_datetime(_input: str = "") -> str:
    """
    获取当前日期和时间
    
    Args:
        _input: 占位符参数（LangChain Tool 调用时必须传入，但这里不使用）
    
    Returns:
        str: 格式化的当前时间字符串
    """
    print("获取当前时间中...")
    # datetime.now() 返回 datetime 对象，需要转换为字符串
    current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    return f"当前时间: {current_time}"


# ----------------------------------------------------------------------------
# 工具6: read_document - 读取 Word 文档工具
# 功能: 读取 .doc 或 .docx 文件的内容
# 使用场景: Agent 需要分析用户上传的 Word 文档时
# ----------------------------------------------------------------------------
def read_document(file_path: str) -> str:
    print("正在读取 Word 文档...")
    print("文件名为："+file_path)
    """
    读取 Word 文档（.doc 或 .docx）的内容

    Args:
        file_path: Word 文档的文件路径（相对路径或绝对路径）

    Returns:
        str: 文档的文本内容

    异常处理:
        如果文件不存在、格式错误或读取失败，返回错误信息
    """
    try:
        if not os.path.exists(file_path):
            return f"错误：文件不存在 - {file_path}"

        if not (file_path.endswith('.doc') or file_path.endswith('.docx')):
            return f"错误：不支持的文件格式，仅支持 .doc 和 .docx 文件"

        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()

        if not documents:
            return f"错误：文档内容为空 - {file_path}"

        content = "\n\n".join([doc.page_content for doc in documents])

        if len(content) > 10000:
            content = content[:10000] + "\n\n[文档内容较长，已截取前10000字符]"

        return f"文档路径: {file_path}\n文档长度: {len(content)} 字符\n\n--- 文档内容 ---\n{content}"

    except Exception as e:
        return f"读取文档失败: {str(e)}"


# ----------------------------------------------------------------------------
# 工具8: read_python_file - 读取 Python 文件工具
# 功能: 读取 .py 文件的完整内容
# 使用场景: Agent 需要查看和分析 Python 代码时
# ----------------------------------------------------------------------------
@tool
def read_python_file(file_path: str) -> str:
    """
    读取 Python 文件的完整内容
    
    Args:
        file_path: Python 文件的路径（相对或绝对路径）
    
    Returns:
        str: 文件的完整内容，包含行号
    """
    print("正在读取 Python 文件...")
    print("文件名为："+file_path)
    try:
        if not os.path.exists(file_path):
            return f"错误：文件不存在 - {file_path}"
        
        if not file_path.endswith('.py'):
            return f"错误：不是 Python 文件（.py）- {file_path}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 添加行号
        numbered_lines = []
        for i, line in enumerate(lines, 1):
            numbered_lines.append(f"{i:4d} | {line.rstrip()}")
        
        content = "\n".join(numbered_lines)
        total_lines = len(lines)
        
        return f"文件: {file_path}\n总行数: {total_lines}\n\n--- 文件内容 ---\n{content}"
    
    except Exception as e:
        return f"读取文件失败: {str(e)}"


# ----------------------------------------------------------------------------
# 工具9: save_python_file - 保存/修改 Python 文件工具
# 功能: 保存修改后的 Python 代码到文件
# 使用场景: Agent 修改代码后保存文件
# ----------------------------------------------------------------------------
@tool
def save_python_file(code: str, file_path: str) -> str:
    """
    保存 Python 代码到文件（覆盖原文件或创建新文件）
    
    Args:
        code: 要保存的 Python 代码内容
        file_path: 保存的文件路径
    
    Returns:
        str: 保存成功的确认信息
    """
    print("正在保存 Python 文件...")
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # 写入文件（覆盖模式）
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # 计算行数
        lines = code.count('\n') + 1
        
        return f"✅ 文件已成功保存: {file_path}\n📊 代码行数: {lines} 行\n💾 文件大小: {len(code.encode('utf-8'))} 字节"
    
    except Exception as e:
        return f"❌ 保存文件失败: {str(e)}"


# ----------------------------------------------------------------------------
# 工具10: analyze_csv_excel - CSV/Excel 数据分析工具
# 功能: 读取 CSV/Excel 文件并生成统计分析报告和图表
# 使用场景: Agent 需要分析数据文件时
# ----------------------------------------------------------------------------
@tool
def analyze_csv_excel(file_path: str, output_dir: str = "analysis_output") -> str:
    """
    读取 CSV 或 Excel 文件，生成统计分析报告和可视化图表
    
    Args:
        file_path: CSV 或 Excel 文件路径
        output_dir: 输出目录（默认: analysis_output）
    
    Returns:
        str: 分析报告摘要和生成的文件列表
    """
    print(f"正在分析文件: {file_path}")
    
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return f"错误：文件不存在 - {file_path}"
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 读取文件
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return f"错误：不支持的文件格式，仅支持 .csv, .xlsx, .xls"
        
        # 生成文件名前缀
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{base_name}_{timestamp}"
        
        # 1. 基本统计信息
        report = []
        report.append("=" * 60)
        report.append(f"数据分析报告")
        report.append(f"文件: {file_path}")
        report.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        report.append("")
        
        # 数据集概览
        report.append("【1. 数据集概览】")
        report.append(f"- 行数: {len(df)}")
        report.append(f"- 列数: {len(df.columns)}")
        report.append(f"- 列名: {', '.join(df.columns.tolist())}")
        report.append("")
        
        # 数据类型
        report.append("【2. 数据类型】")
        for col, dtype in df.dtypes.items():
            report.append(f"- {col}: {dtype}")
        report.append("")
        
        # 缺失值统计
        report.append("【3. 缺失值统计】")
        missing = df.isnull().sum()
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        for col in df.columns:
            if missing[col] > 0:
                report.append(f"- {col}: {missing[col]} ({missing_pct[col]}%)")
        if missing.sum() == 0:
            report.append("- 无缺失值")
        report.append("")
        
        # 数值列统计
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            report.append("【4. 数值列统计】")
            stats = df[numeric_cols].describe()
            for col in numeric_cols:
                report.append(f"\n{col}:")
                report.append(f"  - 均值: {stats[col]['mean']:.2f}")
                report.append(f"  - 标准差: {stats[col]['std']:.2f}")
                report.append(f"  - 最小值: {stats[col]['min']:.2f}")
                report.append(f"  - 最大值: {stats[col]['max']:.2f}")
                report.append(f"  - 中位数: {stats[col]['50%']:.2f}")
            report.append("")
        
        # 分类列统计
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            report.append("【5. 分类列统计】")
            for col in categorical_cols[:5]:  # 只显示前5个分类列
                unique_count = df[col].nunique()
                top_values = df[col].value_counts().head(3)
                report.append(f"\n{col}:")
                report.append(f"  - 唯一值数量: {unique_count}")
                report.append(f"  - 最常见值:")
                for val, count in top_values.items():
                    report.append(f"    • {val}: {count}次")
            report.append("")
        
        # 2. 生成图表
        generated_files = []
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 图表1: 数值列分布直方图
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(min(len(numeric_cols), 3), 1, figsize=(10, 4*min(len(numeric_cols), 3)))
            if len(numeric_cols) == 1:
                axes = [axes]
            
            for i, col in enumerate(numeric_cols[:3]):
                axes[i].hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                axes[i].set_title(f'{col} 分布', fontsize=12, fontweight='bold')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('频数')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            hist_file = os.path.join(output_dir, f"{prefix}_histogram.png")
            plt.savefig(hist_file, dpi=150, bbox_inches='tight')
            plt.close()
            generated_files.append(hist_file)
            report.append("【6. 生成的图表】")
            report.append(f"- 直方图: {hist_file}")
        
        # 图表2: 相关性热力图
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       square=True, linewidths=0.5)
            plt.title('变量相关性热力图', fontsize=14, fontweight='bold')
            plt.tight_layout()
            corr_file = os.path.join(output_dir, f"{prefix}_correlation.png")
            plt.savefig(corr_file, dpi=150, bbox_inches='tight')
            plt.close()
            generated_files.append(corr_file)
            report.append(f"- 相关性热力图: {corr_file}")
        
        # 图表3: 分类列柱状图
        if len(categorical_cols) > 0:
            fig, axes = plt.subplots(min(len(categorical_cols), 2), 1, figsize=(10, 5*min(len(categorical_cols), 2)))
            if len(categorical_cols) == 1:
                axes = [axes]
            
            for i, col in enumerate(categorical_cols[:2]):
                value_counts = df[col].value_counts().head(10)
                axes[i].bar(range(len(value_counts)), value_counts.values, color='coral', edgecolor='black')
                axes[i].set_xticks(range(len(value_counts)))
                axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
                axes[i].set_title(f'{col} 分布', fontsize=12, fontweight='bold')
                axes[i].set_ylabel('频数')
                axes[i].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            bar_file = os.path.join(output_dir, f"{prefix}_bar.png")
            plt.savefig(bar_file, dpi=150, bbox_inches='tight')
            plt.close()
            generated_files.append(bar_file)
            report.append(f"- 柱状图: {bar_file}")
        
        # 3. 保存报告
        report.append("")
        report.append("=" * 60)
        report.append("分析完成！")
        report.append(f"生成的文件保存在: {output_dir}")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # 保存报告到文件
        report_file = os.path.join(output_dir, f"{prefix}_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        generated_files.append(report_file)
        
        # 返回摘要
        summary = f"✅ 分析完成！\n\n"
        summary += f"📊 数据集: {len(df)} 行 × {len(df.columns)} 列\n"
        summary += f"📈 数值列: {len(numeric_cols)} 个\n"
        summary += f"🏷️ 分类列: {len(categorical_cols)} 个\n"
        summary += f"📁 生成文件: {len(generated_files)} 个\n\n"
        summary += "生成的文件:\n"
        for f in generated_files:
            summary += f"  - {f}\n"
        summary += f"\n完整报告已保存至: {report_file}"
        
        return summary
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"❌ 分析失败: {str(e)}\n\n详细错误:\n{error_detail}"
# ----------------------------------------------------------------------------
# 工具7: run_python_code - Python 代码执行工具
# 功能: 安全地执行 Python 代码并返回结果
# 使用场景: Agent 需要运行代码示例、计算、数据处理等
# ----------------------------------------------------------------------------
def run_python_code(code: str) -> str:
    """
    执行 Python 代码或运行 Python 文件
    
    Args:
        code: 要执行的 Python 代码字符串，或者是文件路径（如果以 .py 结尾）
    
    Returns:
        str: 代码执行的输出结果或错误信息
    
    支持的功能:
        - 直接执行 Python 代码字符串
        - 运行 .py 文件（传入文件路径）
        - 导入常用安全模块（math, random, json, datetime 等）
    """
    print("正在运行python代码")
    try:
        # 检查是否是文件路径
        if code.strip().endswith('.py'):
            # 尝试读取并执行 Python 文件
            file_path = code.strip()
            if not os.path.exists(file_path):
                return f"错误：文件不存在 - {file_path}"
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except Exception as e:
                return f"读取文件失败: {str(e)}"
        
        # 创建执行环境，允许导入安全的模块
        safe_globals = {
            '__builtins__': __builtins__,  # 允许所有内置函数
            '__name__': '__main__',
            '__file__': '<executed_code>',
        }
        
        # 预导入常用的安全模块
        safe_modules = [
            'math', 'random', 'datetime', 'json', 'collections',
            'itertools', 'functools', 're', 'string', 'time',
            'copy', 'decimal', 'fractions', 'statistics',
            'typing', 'dataclasses'
        ]
        
        for module_name in safe_modules:
            try:
                module = __import__(module_name)
                safe_globals[module_name] = module
            except ImportError:
                pass
        
        # 捕获标准输出和错误输出
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            # 重定向输出
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # 执行代码
                exec(compile(code, '<code>', 'exec'), safe_globals)
            
            # 获取输出
            stdout = stdout_buffer.getvalue()
            stderr = stderr_buffer.getvalue()
            
            # 组合结果
            result = []
            if stdout:
                result.append("=== 输出结果 ===")
                result.append(stdout.rstrip())
            
            if stderr:
                result.append("=== 错误信息 ===")
                result.append(stderr.rstrip())
            
            if not result:
                return "✅ 代码执行成功，但没有输出"
            
            return "\n\n".join(result)
            
        except Exception as e:
            error_msg = f"❌ 执行错误:\n{type(e).__name__}: {str(e)}"
            # 如果有部分输出，也包含进来
            stdout = stdout_buffer.getvalue()
            if stdout:
                error_msg = f"=== 部分输出 ===\n{stdout}\n\n{error_msg}"
            return error_msg
    
    except Exception as e:
        return f"代码执行失败: {str(e)}"




# ============================================================================
# 第3部分: 工具注册
# ============================================================================
# 将上面定义的函数包装成 LangChain 的 Tool 对象
# Agent 通过 Tool 的 name 和 description 来决定何时调用哪个工具
#
# Tool 类的三个核心参数:
#   - name: 工具名称（唯一标识符，Agent 调用时使用）
#   - func: 实际执行的 Python 函数
#   - description: 工具功能描述（**非常重要**，Agent 根据描述决定是否使用该工具）
# ============================================================================

# ----------------------------------------------------------------------------
# 工具注册 1: save_text_to_file
# 用途: 保存研究数据到本地文件
# 调用时机: Agent 收集完信息后，需要持久化存储时
# ----------------------------------------------------------------------------
save_tool = Tool(
    name="save_text_to_file",    # 工具名称：语义明确，Agent 容易理解
    func=save_to_txt,            # 绑定的函数：上面定义的 save_to_txt
    description="保存搜索结果到本地文件,传入两个参数,第一个为保存内容，第二个为保存的文件名",  # 功能描述：简洁明了
)

# ----------------------------------------------------------------------------
# 工具注册 2: search
# 用途: 搜索互联网获取最新信息
# 调用时机: 用户需要查询新闻、教程、最新资料时
# 描述设计技巧:
#   - 明确搜索源："using DuckDuckGo"（让 Agent 知道搜索引擎）
#   - 提示输入格式："Input: search query string"（指导 Agent 如何传参）
# ----------------------------------------------------------------------------
search_tool = Tool(
    name="search",               # 工具名称：简短有力
    func=web_search,             # 绑定的函数：上面定义的 web_search
    description="Search the web for information using DuckDuckGo. Input: search query string.",
)

# ----------------------------------------------------------------------------
# 工具注册 3: wikipedia
# 用途: 查询权威百科知识
# 调用时机: 用户询问概念定义、背景知识、历史信息等
# 关键词设计:
#   - "encyclopedic": 强调百科全书式的权威信息
#   - "topic or keyword": 提示适合查询主题性知识
# ----------------------------------------------------------------------------
wiki_tool = Tool(
    name="wikipedia",            # 工具名称：直接使用 Wikipedia 这个广为人知的名字
    func=wiki_search,            # 绑定的函数：上面定义的 wiki_search
    description="Search Wikipedia for encyclopedic information. Input: topic or keyword.",
)

exit_tool = Tool(
    name="exit",
    func=exit_agent,
    description="if user say goodbye",
)

time_tool = Tool(
    name="get_time",
    func=get_datetime,
    description="get current time",
)

read_doc_tool = Tool(
    name="read_document",
    func=read_document,
    description="读取Word文档(.doc或.docx)的完整内容。使用方法：传入文件的绝对路径或相对路径。例如：'李健驹论文.docx' 或 'D:/python_ai_agent/mytest/李健驹论文.docx'。此工具会返回文档的全部文本内容，供后续分析和总结使用。",
)

run_code_tool = Tool(
    name="run_python_code",
    func=run_python_code,
    description="执行Python代码或运行.py文件。用法：1) 直接传入代码字符串，如'print(2+2)'；2) 传入.py文件路径，如'test_code_execution.py'。支持导入常用模块(math, random, json, datetime等)。可以执行计算、数据处理、算法演示、文件读取等操作。",
)

# ----------------------------------------------------------------------------
# 工具11: analyze_code_quality - 代码质量分析工具
# 功能: 分析 Python 代码的质量，包括复杂度、风格、潜在问题
# 使用场景: Agent 需要评估代码质量或提供改进建议时
# ----------------------------------------------------------------------------
@tool
def analyze_code_quality(file_path: str) -> str:
    """
    分析 Python 代码的质量，包括：
    1. 代码复杂度（圈复杂度）
    2. 代码风格检查（PEP 8）
    3. 潜在问题和改进建议
    4. 代码行数统计
    
    Args:
        file_path: Python 文件的路径
        
    Returns:
        str: 代码质量分析报告
    """
    print(f"正在分析代码质量: {file_path}")
    
    try:
        if not os.path.exists(file_path):
            return f"错误：文件不存在 - {file_path}"
        
        if not file_path.endswith('.py'):
            return f"错误：不是 Python 文件（.py）- {file_path}"
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
        
        report_lines = []
        report_lines.append(f"📊 代码质量分析报告")
        report_lines.append(f"📁 文件: {file_path}")
        report_lines.append(f"⏰ 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 60)
        
        # 1. 基础统计
        lines = code_content.split('\n')
        total_lines = len(lines)
        non_empty_lines = len([line for line in lines if line.strip()])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        
        report_lines.append("【1. 基础统计】")
        report_lines.append(f"  📈 总行数: {total_lines}")
        report_lines.append(f"  📈 非空行数: {non_empty_lines}")
        report_lines.append(f"  📈 注释行数: {comment_lines}")
        report_lines.append(f"  📈 注释比例: {comment_lines/max(non_empty_lines, 1)*100:.1f}%")
        
        # 2. 函数统计
        import ast
        try:
            tree = ast.parse(code_content)
            
            # 统计函数和类
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            
            report_lines.append("\n【2. 代码结构】")
            report_lines.append(f"  🔧 函数数量: {len(functions)}")
            report_lines.append(f"  🏗️  类数量: {len(classes)}")
            report_lines.append(f"  📦 导入模块: {len(imports)}")
            
            # 函数复杂度分析（简单版）
            if functions:
                report_lines.append("\n【3. 函数分析】")
                for func in functions[:5]:  # 只显示前5个函数
                    func_lines = func.end_lineno - func.lineno if hasattr(func, 'end_lineno') else 0
                    # 简单复杂度估算：嵌套层数
                    complexity = 0
                    for node in ast.walk(func):
                        if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                            complexity += 1
                    
                    report_lines.append(f"  📝 {func.name}:")
                    report_lines.append(f"    行数: {func_lines}, 复杂度: {complexity}")
                    
                    if complexity > 5:
                        report_lines.append(f"    ⚠️  建议: 函数 {func.name} 复杂度较高，考虑拆分")
                    elif func_lines > 50:
                        report_lines.append(f"    ⚠️  建议: 函数 {func.name} 过长，考虑重构")
        
        except SyntaxError as e:
            report_lines.append(f"\n⚠️ 语法错误: {str(e)}")
        
        # 3. 代码风格检查（简单版）
        report_lines.append("\n【4. 代码风格检查】")
        
        # 检查行长
        long_lines = [(i+1, line) for i, line in enumerate(lines) if len(line) > 79]
        if long_lines:
            report_lines.append(f"  ⚠️  发现 {len(long_lines)} 行超过79字符:")
            for line_num, line in long_lines[:3]:  # 只显示前3个
                report_lines.append(f"    第{line_num}行: {line[:50]}...")
        
        # 检查导入顺序
        import_sections = []
        current_section = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                current_section.append(line)
            elif current_section:
                import_sections.append(current_section)
                current_section = []
        
        if len(import_sections) > 1:
            report_lines.append(f"  ⚠️  导入语句被代码分隔，建议集中放置")
        
        # 4. 潜在问题检查
        report_lines.append("\n【5. 潜在问题】")
        
        # 检查硬编码字符串
        hardcoded_strings = []
        for i, line in enumerate(lines):
            if 'password' in line.lower() or 'secret' in line.lower() or 'key' in line.lower():
                if '"' in line or "'" in line:
                    hardcoded_strings.append((i+1, line.strip()[:50]))
        
        if hardcoded_strings:
            report_lines.append(f"  🔒 发现可能硬编码的敏感信息:")
            for line_num, line in hardcoded_strings[:3]:
                report_lines.append(f"    第{line_num}行: {line}...")
        
        # 检查TODO/FIXME
        todo_lines = [(i+1, line.strip()) for i, line in enumerate(lines) 
                     if 'TODO' in line.upper() or 'FIXME' in line.upper()]
        if todo_lines:
            report_lines.append(f"  📋 发现 {len(todo_lines)} 个待办事项:")
            for line_num, line in todo_lines[:3]:
                report_lines.append(f"    第{line_num}行: {line}")
        
        # 5. 改进建议
        report_lines.append("\n【6. 改进建议】")
        
        suggestions = []
        if comment_lines / max(non_empty_lines, 1) < 0.1:
            suggestions.append("增加注释，提高代码可读性")
        
        if len(long_lines) > 5:
            suggestions.append("缩短过长的代码行（>79字符）")
        
        if len(functions) > 0 and any(len(functions) > 10 for _ in functions):
            suggestions.append("考虑将大型函数拆分为更小的函数")
        
        if not suggestions:
            suggestions.append("代码质量良好，继续保持！")
        
        for i, suggestion in enumerate(suggestions, 1):
            report_lines.append(f"  {i}. {suggestion}")
        
        report_lines.append("\n" + "=" * 60)
        report_lines.append("💡 提示: 使用更专业的工具（如 pylint、flake8）进行详细分析")
        
        return "\n".join(report_lines)
        
    except Exception as e:
        return f"代码质量分析失败: {str(e)}"


# 不需要手动注册，@tool 装饰器已经处理
# read_py_file_tool 和 save_py_file_tool 已经在上面通过 @tool 装饰器创建


# ============================================================================
# 向量数据库管理（全局变量）
# ============================================================================
_vector_db = None
_embeddings = None
_db_initialized = False


def _get_or_create_vector_db(persist_directory: str = "vector_db"):
    """
    获取或创建向量数据库实例（单例模式）
    
    Args:
        persist_directory: 向量数据库持久化目录
    
    Returns:
        Chroma: 向量数据库实例
    """
    global _vector_db, _embeddings, _db_initialized
    
    if not _db_initialized:
        try:
            # 设置环境变量以禁用警告
            import os
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            
            # 初始化嵌入模型（使用中文模型）
            print("正在加载嵌入模型（首次使用需要下载约400MB）...")
            _embeddings = HuggingFaceEmbeddings(
                model_name="shibing624/text2vec-base-chinese",
                model_kwargs={'device': 'cpu'}  # 使用 CPU，如果有 GPU 可以改为 'cuda'
            )
            
            # 创建或加载向量数据库
            print(f"正在初始化向量数据库: {persist_directory}")
            _vector_db = Chroma(
                embedding_function=_embeddings,
                persist_directory=persist_directory
            )
            
            _db_initialized = True
            print("✅ 向量数据库初始化成功")
            
        except Exception as e:
            print(f"❌ 向量数据库初始化失败: {e}")
            raise
    
    return _vector_db


# ----------------------------------------------------------------------------
# 工具13: add_documents_to_vector_db - 添加文档到向量数据库
# 功能: 将文本内容添加到向量数据库中，支持语义搜索
# 使用场景: 建立知识库、文档检索系统
# ----------------------------------------------------------------------------
@tool
def add_documents_to_vector_db(texts: str, collection_name: str = "default", metadata: str = "") -> str:
    """
    将一个或多个文档添加到向量数据库中
    
    Args:
        texts: 要添加的文本内容，多个文档用 '|||' 分隔
        collection_name: 集合名称（用于分类存储不同主题的文档）
        metadata: 元数据信息（JSON格式字符串，可选）
    
    Returns:
        str: 添加成功的确认信息
    
    示例:
        # 添加单个文档
        add_documents_to_vector_db("Python是一种编程语言", "programming")
        
        # 添加多个文档
        add_documents_to_vector_db(
            "文档1内容|||文档2内容|||文档3内容",
            "knowledge_base",
            '{"source": "manual", "date": "2024-01-01"}'
        )
    """
    print(f"正在添加文档到向量数据库 (集合: {collection_name})...")
    
    try:
        # 获取或创建向量数据库
        vector_db = _get_or_create_vector_db()
        
        # 分割多个文档
        text_list = [t.strip() for t in texts.split('|||') if t.strip()]
        
        if not text_list:
            return "错误：没有提供有效的文本内容"
        
        # 解析元数据
        meta_dict = {}
        if metadata:
            try:
                import json
                meta_dict = json.loads(metadata)
            except:
                meta_dict = {"info": metadata}
        
        # 创建 Document 对象列表
        documents = []
        for i, text in enumerate(text_list):
            doc_metadata = meta_dict.copy()
            doc_metadata["index"] = i
            doc_metadata["collection"] = collection_name
            doc_metadata["timestamp"] = datetime.now().isoformat()
            
            documents.append(Document(
                page_content=text,
                metadata=doc_metadata
            ))
        
        # 添加到向量数据库
        vector_db.add_documents(documents)
        
        return f"✅ 成功添加 {len(documents)} 个文档到集合 '{collection_name}'\n📊 文档总数量: {vector_db._collection.count()}"
    
    except Exception as e:
        import traceback
        return f"❌ 添加文档失败: {str(e)}\n\n{traceback.format_exc()}"


# ----------------------------------------------------------------------------
# 工具14: search_vector_db - 向量数据库语义搜索
# 功能: 基于语义相似度搜索相关文档
# 使用场景: 知识检索、问答系统、文档推荐
# ----------------------------------------------------------------------------
@tool
def search_vector_db(query: str, collection_name: str = "default", top_k: int = 5) -> str:
    """
    在向量数据库中搜索与查询最相关的文档
    
    Args:
        query: 搜索查询文本
        collection_name: 要搜索的集合名称
        top_k: 返回最相关的 K 个结果（默认5个）
    
    Returns:
        str: 搜索结果，包含文档内容和相似度分数
    
    示例:
        search_vector_db("如何学习Python?", "programming", 3)
    """
    print(f"正在向量数据库中搜索: {query}")
    
    try:
        # 获取向量数据库
        vector_db = _get_or_create_vector_db()
        
        # 执行相似性搜索
        results = vector_db.similarity_search_with_score(
            query=query,
            k=top_k
        )
        
        if not results:
            return f"⚠️ 在集合 '{collection_name}' 中未找到相关文档"
        
        # 格式化结果
        output = [f"🔍 搜索结果 (共 {len(results)} 条):\n"]
        output.append(f"查询: {query}\n")
        output.append("=" * 60)
        
        for i, (doc, score) in enumerate(results, 1):
            similarity = 1 - score  # 转换为相似度（0-1之间）
            output.append(f"\n【结果 {i}】 相似度: {similarity:.4f}")
            output.append(f"内容: {doc.page_content[:500]}")  # 限制显示长度
            
            if doc.metadata:
                output.append(f"元数据: {doc.metadata}")
            
            output.append("-" * 60)
        
        return "\n".join(output)
    
    except Exception as e:
        import traceback
        return f"❌ 搜索失败: {str(e)}\n\n{traceback.format_exc()}"


# ----------------------------------------------------------------------------
# 工具15: clear_vector_db - 清空向量数据库
# 功能: 删除向量数据库中的所有文档
# 使用场景: 重置知识库、清理测试数据
# ----------------------------------------------------------------------------
@tool
def clear_vector_db(collection_name: str = "default") -> str:
    """
    清空指定集合中的所有文档
    
    Args:
        collection_name: 要清空的集合名称
    
    Returns:
        str: 操作结果
    """
    print(f"正在清空集合: {collection_name}")
    
    try:
        vector_db = _get_or_create_vector_db()
        
        # 获取当前文档数量
        count_before = vector_db._collection.count()
        
        # 删除所有文档
        vector_db._collection.delete(where={"collection": collection_name})
        
        count_after = vector_db._collection.count()
        deleted = count_before - count_after
        
        return f"✅ 已清空集合 '{collection_name}'\n🗑️ 删除文档数: {deleted}\n📊 剩余文档数: {count_after}"
    
    except Exception as e:
        return f"❌ 清空失败: {str(e)}"


# ============================================================================
# 用户偏好管理系统
# ============================================================================
_user_preferences_file = "user_preferences.json"

def _load_user_preferences() -> dict:
    """加载用户偏好"""
    if os.path.exists(_user_preferences_file):
        try:
            with open(_user_preferences_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def _save_user_preferences(preferences: dict):
    """保存用户偏好"""
    try:
        with open(_user_preferences_file, 'w', encoding='utf-8') as f:
            json.dump(preferences, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存偏好失败: {e}")
        return False


# ----------------------------------------------------------------------------
# 工具16: save_user_preference - 保存用户偏好
# 功能: 记录用户的编程习惯、语言偏好、工具选择等
# 使用场景: Agent 学习用户习惯，提供个性化服务
# ----------------------------------------------------------------------------
@tool
def save_user_preference(category: str, key: str, value: str) -> str:
    """
    保存用户的偏好设置
    
    Args:
        category: 偏好类别（如：'coding_style', 'language', 'tools_file', 'workflow'）
        key: 偏好键名（如：'indentation', 'preferred_language', 'testing_framework'）
        value: 偏好值（如：'4 spaces', 'Python', 'pytest'）
    
    Returns:
        str: 保存成功的确认信息
    
    示例:
        save_user_preference("coding_style", "indentation", "4 spaces")
        save_user_preference("language", "response_language", "中文")
        save_user_preference("tools_file", "testing_framework", "pytest")
    """
    print(f"正在保存用户偏好: {category}.{key} = {value}")
    
    try:
        preferences = _load_user_preferences()
        
        # 确保类别存在
        if category not in preferences:
            preferences[category] = {}
        
        # 保存偏好
        preferences[category][key] = {
            "value": value,
            "updated_at": datetime.now().isoformat()
        }
        
        # 保存到文件
        if _save_user_preferences(preferences):
            return f"✅ 已保存偏好: {category}.{key} = {value}"
        else:
            return "❌ 保存失败"
    
    except Exception as e:
        return f"❌ 保存偏好失败: {str(e)}"


# ----------------------------------------------------------------------------
# 工具17: get_user_preferences - 获取用户偏好
# 功能: 查询用户的所有或特定偏好设置
# 使用场景: Agent 根据用户偏好调整行为
# ----------------------------------------------------------------------------
@tool
def get_user_preferences(category: str = "all") -> str:
    """
    获取用户的偏好设置
    
    Args:
        category: 偏好类别（'all' 表示所有类别，或指定具体类别）
    
    Returns:
        str: 用户偏好设置的文本描述
    
    示例:
        get_user_preferences("all")  # 获取所有偏好
        get_user_preferences("coding_style")  # 获取编码风格偏好
    """
    print(f"正在获取用户偏好: {category}")
    
    try:
        preferences = _load_user_preferences()
        
        if not preferences:
            return "⚠️ 还没有保存任何用户偏好"
        
        output = ["📋 用户偏好设置:\n"]
        output.append("=" * 60)
        
        if category == "all":
            # 显示所有类别
            for cat, prefs in preferences.items():
                output.append(f"\n【{cat}】")
                for key, pref_data in prefs.items():
                    value = pref_data.get("value", "N/A")
                    updated = pref_data.get("updated_at", "Unknown")
                    output.append(f"  • {key}: {value}")
                    output.append(f"    (更新时间: {updated})")
        else:
            # 显示指定类别
            if category in preferences:
                output.append(f"\n【{category}】")
                for key, pref_data in preferences[category].items():
                    value = pref_data.get("value", "N/A")
                    updated = pref_data.get("updated_at", "Unknown")
                    output.append(f"  • {key}: {value}")
                    output.append(f"    (更新时间: {updated})")
            else:
                return f"⚠️ 类别 '{category}' 不存在"
        
        output.append("\n" + "=" * 60)
        return "\n".join(output)
    
    except Exception as e:
        return f"❌ 获取偏好失败: {str(e)}"


# ----------------------------------------------------------------------------
# 工具18: delete_user_preference - 删除用户偏好
# 功能: 删除指定的用户偏好设置
# 使用场景: 清理过时或不需要的偏好
# ----------------------------------------------------------------------------
@tool
def delete_user_preference(category: str, key: str) -> str:
    """
    删除用户的偏好设置
    
    Args:
        category: 偏好类别
        key: 偏好键名
    
    Returns:
        str: 操作结果
    
    示例:
        delete_user_preference("coding_style", "indentation")
    """
    print(f"正在删除用户偏好: {category}.{key}")
    
    try:
        preferences = _load_user_preferences()
        
        if category not in preferences or key not in preferences[category]:
            return f"⚠️ 偏好不存在: {category}.{key}"
        
        # 删除偏好
        deleted_value = preferences[category].pop(key)["value"]
        
        # 如果类别为空，删除整个类别
        if not preferences[category]:
            del preferences[category]
        
        # 保存
        if _save_user_preferences(preferences):
            return f"✅ 已删除偏好: {category}.{key} = {deleted_value}"
        else:
            return "❌ 删除失败"
    
    except Exception as e:
        return f"❌ 删除偏好失败: {str(e)}"



