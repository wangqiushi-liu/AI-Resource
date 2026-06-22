import base64
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pandas as pd
from openai import OpenAI

# ==================== 配置区域 ====================
IMAGES_DIR = "./images"
INPUT_CSV = "./images/action_triples.csv"
OUTPUT_FILE = "check_results.xlsx"
MAX_WORKERS = 5  # 并发数，可调整

# ==================== Kimi API 配置 ====================
# TODO: 请在此处填入你的 Kimi API Key
KIMI_API_KEY = ""

extra_body = {
    "chat_template_kwargs": {"enable_thinking": True}
}

transport = httpx.HTTPTransport(proxy=None, verify=False)
http_client = httpx.Client(transport=transport)

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=KIMI_API_KEY,
    http_client=http_client
)


def encode_image(image_path):
    """将图片编码为 base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_local_image_path(csv_path):
    """
    将 CSV 中的 Windows 路径转换为本地图片路径
    CSV 路径示例: D:\\code\\UIAutoTest2\\runs\\run_20260604_212327\\screenshots\\000000_0003_d1d5d5c7a5c9e1ae.png
    本地路径示例: ./images/000000_0003_d1d5d5c7a5c9e1ae.png
    """
    # CSV 路径只取最后一段文件名，兼容 Windows 和 macOS 路径分隔符
    # 统一路径分隔符后取文件名，兼容 Windows 和 macOS
    normalized = csv_path.replace('\\', '/')
    filename = normalized.split('/')[-1]
    return os.path.join(IMAGES_DIR, filename)


def call_kimi_check(before_path, action, after_path):
    """
    调用 Kimi 模型检查图片三元组
    before_path: 操作前的图片路径
    action: 遥控器操作 (如 DPAD_DOWN)
    after_path: 操作后的图片路径
    """
    before = encode_image(before_path)
    after = encode_image(after_path)

    prompt = f'''
你是一个严格的 EPG (电子节目指南) 自动化测试工程师，负责发现 IPTV 系统的各类视觉与逻辑缺陷。你的核心原则是：**宁可误报，不可漏报**——任何可疑的、不确定的、边界模糊的情况，都应判为 FAIL。

## 任务
请分析以下两张图片：
- 【第一张图 (pre)】：执行操作前的当前状态。
- 【第二张图 (post)】：在 pre 状态下，用户按遥控器执行了 "{action}" 后的实际结果。

## 遥控器操作说明
- DPAD_UP：向上按钮
- DPAD_DOWN：向下按钮
- DPAD_LEFT：向左按钮
- DPAD_RIGHT：向右按钮
- DPAD_CENTER：确认按钮
- BACK：返回按钮

## 第一阶段：预期逻辑推理
在查看 post 图片前，请先独立判断：执行 "{action}" 后的**标准预期结果**应该是什么？
- 如果是"返回"，应该跳转逻辑上的上一层页面。
- 如果是"确定"，应该进入视频播放器或详情页。
- 如果是"方向键"，且该方向有移动空间的情况下，**焦点必须发生移动，且旧焦点必须消失**。

## 第二阶段：缺陷判定标准（必须逐项检查）

### A. 逻辑一致性缺陷
1. **操作无效**：执行操作后，界面与 pre 完全一致 → FAIL
2. **跳转错误**：操作（如"进入直播"）与 post 内容（如"留在设置页"）不符 → FAIL
3. **旧焦点残留**：焦点移动了，但旧的焦点高亮没有消失 → FAIL
4. **焦点移动错误**：焦点移动方向与按下的键不符（如按 DOWN 但焦点向上移动）→ FAIL
5. **焦点移动距离异常**：焦点跳过多个选项、或移动距离明显不合理 → FAIL

### B. 渲染污染缺陷
6. **元素重叠**：`post` 出现了本该消失的 `pre` 界面元素 → FAIL
7. **残影/鬼影**：文字、图片位移后原位置留有残影 → FAIL
8. **UI 组件堆叠**：多个 UI 组件在同一位置堆叠导致无法辨认 → FAIL

### C. 核心功能区缺陷
9. **局部黑屏**：UI 框架加载成功，但核心内容区（视频播放窗口、海报位）呈纯黑色 → FAIL
10. **焦点丢失**：操作后屏幕上找不到任何高亮选框 → FAIL
11. **多重焦点**：屏幕上同时出现两个或以上高亮选框 → FAIL

### D. 视觉崩溃缺陷
12. **全屏黑屏 / 花屏 / 乱码** → FAIL
13. **文字重叠**导致视觉混乱 → FAIL

### E. 极易漏报的边界情况（重点检查）
14. **轻微残影**：残影不明显，只在仔细对比时才能发现 → FAIL（漏报高危区）
15. **次要元素错位**：背景、次要文字轻微错位但主要焦点看似正常 → FAIL
16. **加载不完整**：内容显示了一部分但明显不完整（如列表只加载了一半）→ FAIL
17. **色彩/亮度异常**：整体或局部色彩/亮度与预期有细微差异 → FAIL
18. **动画/过渡未完成**：画面像是卡在了某个中间状态 → FAIL

## 第三阶段：客观指标检查
- **文件大小**: 若 post 比 pre 文件减小 >30%，高度警惕黑屏 → FAIL
- **分辨率**: 若 post 分辨率异常 → FAIL

## 第四阶段：PASS 判定条件（必须全部满足）
只有满足以下**所有条件**才能判为 PASS，缺一不可：
1. 焦点移动**方向正确**（与操作键一致）
2. 焦点移动**有实际位移**（不能停在原位）
3. 旧焦点**完全消失**（不能残留）
4. 界面**没有新出现的缺陷**（无残影、无重叠、无黑屏）
5. 内容变化**符合操作语义**（如按确定应该进入详情页，而非跳到无关页面）

## 第五阶段：特殊场景处理（严格限制）
以下情况**不能**直接判为 PASS，必须进一步分析：
- 画面无变化 → 必须确认：该方向**确实**没有操作空间，且没有循环现象
- 循环导航（如最下再 DOWN 跳到最上）→ 必须确认：这是该列表的**设计如此**还是异常行为
- 不确定时 → **直接判为 FAIL**，不要尝试"合理化"异常现象

## 输出要求
请按以下 JSON 格式输出，**description 字段必须详细说明判断依据**：

```json
{{
  "logical_consistency": "一致/不一致",
  "defect_types": ["A.1", "B.7"...],  // 发现的缺陷类型编号，未发现则为空数组
  "description": "详细的判断依据，必须说明：1.预期是什么 2.实际看到了什么 3.差异在哪里",
  "overall_verdict": "PASS或FAIL",
  "confidence": 0.0到1.0
}}
```
'''

    try:
        completion = client.chat.completions.create(
            model="kimi-k2.6",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{before}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{after}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            stream=False,
            extra_body=extra_body,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"API 调用错误: {e}")
        return ""


def parse_model_output(raw_output):
    """解析模型输出，提取 overall_verdict、description 和 defect_types"""
    try:
        cleaned = raw_output.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        data = json.loads(cleaned)
        verdict = data.get("overall_verdict", "Unknown")
        description = data.get("description", "")
        defect_types = data.get("defect_types", [])
        return verdict, description, defect_types
    except Exception as e:
        print(f"解析 JSON 失败: {e}")
        return "Parse Error", raw_output, []


def process_row(row):
    """处理单行数据"""
    before_csv_path = row['before_image']
    action = row['action']
    after_csv_path = row['after_image']

    before_local = get_local_image_path(before_csv_path)
    after_local = get_local_image_path(after_csv_path)

    if not os.path.exists(before_local):
        print(f"图片不存在: {before_local}")
        return None
    if not os.path.exists(after_local):
        print(f"图片不存在: {after_local}")
        return None

    print(f"处理: {os.path.basename(before_local)} + {action} -> {os.path.basename(after_local)}")

    raw_result = call_kimi_check(before_local, action, after_local)
    verdict, description, defect_types = parse_model_output(raw_result)

    return {
        "before_image": os.path.basename(before_local),
        "action": action,
        "after_image": os.path.basename(after_local),
        "res": verdict,
        "description": description,
        "defect_types": ",".join(defect_types) if defect_types else ""
    }


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"错误: 找不到 CSV 文件 '{INPUT_CSV}'")
        return

    if KIMI_API_KEY == "YOUR_API_KEY_HERE":
        print("错误: 请先在代码中填入你的 Kimi API Key")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"共读取 {len(df)} 条记录")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, row): idx for idx, row in df.iterrows()}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"处理异常: {e}")

    if results:
        result_df = pd.DataFrame(results)
        result_df.to_excel(OUTPUT_FILE, index=False)
        print(f"\n处理完成！结果已保存至: {OUTPUT_FILE}")
    else:
        print("\n未生成任何结果")


if __name__ == "__main__":
    main()