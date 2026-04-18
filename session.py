from datetime import datetime
import os

# Session 文件存储目录
SESSION_DIR = "session_file"
session_data = []
# 确保目录存在
if not os.path.exists(SESSION_DIR):
    os.makedirs(SESSION_DIR)
    print(f"✅ 创建 session 目录: {SESSION_DIR}")


def save_session_to_txt(data: str, filename: str = None):
    """
    保存 session 数据到文本文件
    
    Args:
        data: 要保存的数据
        filename: 文件名（可选，不提供则自动生成）
    
    Returns:
        str: 保存结果信息
    """
    try:
        # 如果没有提供文件名，则根据当前时间自动生成唯一文件名
        if filename is None or filename.strip() == "":
            # 生成格式：session_年-月-日_时-分-秒.txt
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"session_{timestamp}.txt"
            print(f"✅ 自动生成文件名：{filename}")
        
        # 构建完整路径（保存到 session_file 目录）
        filepath = os.path.join(SESSION_DIR, filename)

        # 以追加模式打开文件并写入数据
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(data)
        
        print(f"✅ Session 保存到 {filepath} 成功\n")
        return f"Data successfully saved to {filepath}"
    
    except Exception as e:
        error_msg = f"❌ 保存失败：{str(e)}"
        print(error_msg)
        return error_msg


def load_session_from_txt(filename: str):
    """
    从文本文件加载 session 数据
    
    Args:
        filename: 文件名
    
    Returns:
        str: 文件内容，失败返回 None
    """
    try:
        # 构建完整路径
        filepath = os.path.join(SESSION_DIR, filename)
        
        if not os.path.exists(filepath):
            error_msg = f"❌ 文件不存在：{filepath}"
            print(error_msg)
            return None
        
        # 读取文件内容并返回
        with open(filepath, "r", encoding="utf-8") as f:
            data = f.read()
        
        print(f"✅ Session 从 {filepath} 读取成功\n")
        return data
    
    except Exception as e:
        error_msg = f"❌ 读取失败：{str(e)}"
        print(error_msg)
        return None


def menu(messages_history: list = None):
    """
    Session 管理菜单
    
    Args:
        messages_history: 消息历史记录（用于保存）
    
    Returns:
        list: 更新后的消息历史
    """
    session_data = messages_history if messages_history else []
    
    print("\n" + "=" * 60)
    print("📋 Session 管理菜单")
    print("=" * 60)
    print("1. 保存当前会话")
    print("2. 读取会话文件")
    print("3. 查看session列表")
    print("4. 删除会话文件")
    print("5. 创建新会话文件")
    print("6. 退出菜单")
    print("=" * 60)
    
    choice = input("请选择操作 (1-6): ").strip()
    
    if choice == "1":
        # 保存当前会话
        if messages_history and len(messages_history) > 0:
            # 将消息历史转换为字符串
            data = "\n".join([f"{role}: {content}" for role, content in messages_history])
            filename = input("请输入保存的文件名（直接回车自动生成）：").strip()
            result = save_session_to_txt(data, filename if filename else None)
            print(result)
            return session_data
        else:
            print("⚠️ 当前没有会话记录可保存")
            return session_data
    
    elif choice == "2":
        # 读取会话文件
        filename = input("请输入要读取的文件名：").strip()
        if filename:
            data = load_session_from_txt(filename)
            if data is not None:
                print(f"📄 文件内容长度：{len(data)} 字符")
                # 将读取的文本转换回元组列表格式
                lines = [line.strip() for line in data.split("\n") if line.strip()]
                parsed_data = []
                for line in lines:
                    if ": " in line:
                        # 分割 role 和 content（只分割第一个冒号+空格）
                        parts = line.split(": ", 1)
                        if len(parts) == 2:
                            role, content = parts
                            parsed_data.append((role, content))
                
                if parsed_data:
                    session_data = parsed_data
                    print(f"✅ 成功加载 {len(parsed_data)} 条消息")
                else:
                    print("⚠️ 文件格式不正确，无法解析")
        return session_data
    
    elif choice == "3":
        for filename in os.listdir(SESSION_DIR):
            filepath = os.path.join(SESSION_DIR, filename)
            if os.path.isfile(filepath):
                print(filename)

        return session_data
    
    elif choice == "4":
        # 删除会话文件
        filename = input("请输入要删除的文件名：").strip()
        if filename:
            try:
                filepath = os.path.join(SESSION_DIR, filename)
                if os.path.exists(filepath):
                    confirm = input(f"确认删除 {filepath}？(y/n): ").strip().lower()
                    if confirm == 'y':
                        os.remove(filepath)
                        print(f"✅ 已删除：{filepath}")
                    else:
                        print("❌ 取消删除")
                else:
                    print(f"❌ 文件不存在：{filepath}")
            except Exception as e:
                print(f"❌ 删除失败：{str(e)}")
        return session_data
    
    elif choice == "5":
        # 创建新会话文件
        filename = input("请输入要创建的文件名：").strip()
        if filename:
            result = save_session_to_txt("", filename)
            print(result)
            # 返回空列表，开始新的会话
            return []
        else:
            print("❌ 文件名不能为空")
            return session_data
    
    elif choice == "6":
        # 退出
        print("👋 退出菜单\n")
        return session_data
    
    else:
        print("❌ 无效选择，请输入 1-6\n")
        return session_data

