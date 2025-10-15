import os
import json
import random
import uuid
import requests

# --- 基础配置 ---
# 请根据实际平台（如 OpenAI、DeepSeek 等）替换为真实的 URL 和模型名
API_URL = os.getenv("LLM_API_URL", "https://your-llm-provider.com/v1/chat/completions")
API_KEY = os.getenv("LLM_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "the-model-you-want-to-use")  # 例如 "deepseek-chat"

def call_llm(messages):
    """
    调用大语言模型 API 的示例函数。
    注意：需根据实际使用的 API 规范调整字段。
    """
    if not API_KEY:
        print("错误：缺少环境变量 LLM_API_KEY，请安全配置后再运行。")
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 1024,  # 根据模型限制调整
        "temperature": 0.7
        # 仅当服务商支持时再设置：
        # "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        response_json = response.json()
        # 不同服务商的返回结构可能不同，需按文档解析
        content = response_json['choices'][0]['message']['content']
        return content
    except requests.exceptions.RequestException as e:
        print(f"错误: API 请求失败: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"错误: 解析 API 响应失败，响应结构可能不符合预期: {e}")
        try:
            print(f"原始响应: {response.text}")
        except Exception:
            pass
        return None

def load_seed_examples(filepath, num_examples=3):
    """从初始文件中加载种子示例"""
    seeds = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 随机选择若干个示例，以增加多样性
            selected_lines = random.sample(lines, min(len(lines), num_examples))
            for line in selected_lines:
                if line.strip():
                    seeds.append(json.loads(line))
        return seeds
    except FileNotFoundError:
        print(f"错误: 种子文件 '{filepath}' 未找到。")
        return []
    except json.JSONDecodeError:
        print(f"错误: 种子文件 '{filepath}' 格式不正确，无法解析为 JSON。")
        return []

def build_system_prompt(seed_examples, constraint_list):
    """构建包含 few-shot 示例的系统提示词"""
    prompt = (
        "你是一名资深的测试数据生成专家，擅长为大语言模型设计高质量、包含多种约束的测试用例。"
        "你的任务是根据用户的需求，生成符合特定 JSON 格式的测试数据。\n\n"
    )
    prompt += "--- 以下是若干个高质量的示例，请模仿它们的结构与规范 ---\n\n"

    for i, example in enumerate(seed_examples):
        # 简化示例，仅保留 user 消息与 condition
        simplified_example = {
            "messages": [
                msg for msg in example.get("messages", []) if msg.get("role") == "user"
            ],
            "condition": example.get("condition", [])
        }
        prompt += f"--- 示例 {i+1} ---\n"
        prompt += "```json\n"
        prompt += json.dumps(simplified_example, ensure_ascii=False, indent=2)
        prompt += "\n```\n\n"

    prompt += "--- 约束类型列表 ---\n"
    prompt += "在生成 `condition` 字段时，`constraint_type` 必须从如下列表中选择：\n"
    prompt += f"{', '.join(constraint_list)}\n\n"
    prompt += "--- 任务要求 ---\n"
    prompt += (
        "现在请根据我之后的要求，生成一个全新的、内容不重复但结构与示例一致的测试用例。"
        "*你必须只输出一个完整、可被直接解析的 JSON 对象，不要包含解释性文字或 ```json 标记*"
    )
    return prompt

def generate_synthetic_data(original_data_path, output_path, num_to_generate=50):
    """数据合成主函数"""
    # 约束类型列表（可按需扩充或细化）
    all_constraints = [
        "语义约束", "格式约束", "风格约束", "数值约束", "长度约束", "中文约束",
        "英文约束", "其他语言约束", "示例约束", "专业术语约束", "情感倾向约束",
        "原文约束", "符号约束", "词汇约束", "集合约束", "文本结构约束",
        "时间约束", "主题约束", "结构约束", "流程约束", "边界约束", "其他约束"
    ]

    # 1. 加载种子示例
    seed_examples = load_seed_examples(original_data_path, num_examples=3)
    if not seed_examples:
        print("无法加载种子示例，流程终止。")
        return

    # 2. 构建系统提示词
    system_prompt = build_system_prompt(seed_examples, all_constraints)

    # 3. 循环生成数据
    generated_data_count = 0
    with open(output_path, 'w', encoding='utf-8') as f_out:
        while generated_data_count < num_to_generate:
            print(f"\\n--- 正在生成第 {generated_data_count + 1}/{num_to_generate} 条数据 ---")

            # 随机选择两种不同的约束
            constraint_a, constraint_b = random.sample(all_constraints, 2)

            # 构建用户提示
            user_prompt = (
                f"请生成一个新的测试用例，需同时包含 '{constraint_a}' 和 '{constraint_b}' 两种约束。"
                "确保问题的复杂度与专业性，并使其具有可验证性。"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # 调用 LLM
            llm_response_str = call_llm(messages)

            if llm_response_str:
                try:
                    # 解析返回的 JSON
                    new_data = json.loads(llm_response_str)

                    # 验证基本结构
                    if "messages" in new_data and "condition" in new_data:
                        new_data['id'] = str(uuid.uuid4())  # 赋予新的唯一ID

                        # 写入文件
                        f_out.write(json.dumps(new_data, ensure_ascii=False) + '\\n')
                        f_out.flush()  # 立即写入磁盘

                        generated_data_count += 1
                        print(f"成功：第 {generated_data_count} 条数据已生成并保存。")
                    else:
                        print("警告：LLM 返回的 JSON 结构不完整，缺少 'messages' 或 'condition' 字段，已跳过。")
                except json.JSONDecodeError:
                    print("警告：LLM 返回内容不是有效的 JSON 格式，已跳过。")
                    print(f"收到的无效内容: {llm_response_str}")
            else:
                print("警告：调用 LLM 失败，无法获取响应，已跳过本次生成。")

    print(f"\\n处理完成，共生成 {generated_data_count} 条合成数据，已保存至 '{output_path}'。")

# --- 程序入口 ---
if __name__ == '__main__':
    # 确保你的初始数据文件 '10.9.json' 在该脚本同一目录下
    INPUT_SEED_FILE = '10.9.json'
    OUTPUT_SYNTHETIC_FILE = 'synthetic_dataset.jsonl'

    generate_synthetic_data(INPUT_SEED_FILE, OUTPUT_SYNTHETIC_FILE, num_to_generate=50)
