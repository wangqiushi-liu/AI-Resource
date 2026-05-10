#!/usr/bin/env python3
"""
EPG Visual Defect Detection - 使用 MiniMax 多模态大模型检测 EPG 界面缺陷

输入: pre.png (操作前) + post.png (操作后)  两张图片一次性传入
检测: 界面是否存在黑屏、乱码、花屏、报错等视觉缺陷

使用方式:
    python epg_defect_check.py --pre /path/to/pre.png --post /path/to/post.png

作者: Claude
"""

import argparse
import base64
import json
import os
import sys
from typing import Optional

# ============ MiniMax API 配置 ============
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")
MINIMAX_BASE_URL = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.chat")
MINIMAX_MODEL = "MiniMax-M2.7"

# ============ 检测 Prompt ============
DEFECT_DETECTION_PROMPT = """你是一个专业的 EPG (电子节目指南) 自动化测试工程师。
请分析这两张图片，它们是 IPTV 电视机界面的截图。

【第一张图 (pre)】：操作前的界面截图
【第二张图 (post)】：按遥控器按钮操作后的界面截图

## 核心原则：pre 与 post 是关联的

pre 和 post 是**同一操作序列的先后状态**。你必须对比两张图的关联性：
- post 是用户按遥控器后应该出现的**正常结果**
- 如果 post 出现了**不应该出现的异常状态**，才是缺陷

## 缺陷判断标准（必须严格遵循）

**只有当界面出现以下明确、明显的视觉异常时，才判定为缺陷：**

1. **黑屏/白屏**: 整个或90%以上屏幕为纯黑或纯白色，无任何UI元素可见
2. **花屏/雪花**: 屏幕大量雪花点、彩色条纹、严重的像素化或色彩失真，无法正常观看内容
3. **乱码**: 文字完全无法辨认，变成符号或方块，正常的语言文字消失
4. **UI 严重错位**: 界面元素明显重叠、遮挡、或完全偏离正常位置，导致内容不可读
5. **焦点丢失**: 正常应该有一个高亮选框（蓝色边框或背景色）但完全消失，其他UI正常
6. **明确的报错弹窗**: 必须同时满足以下条件：
   - 有明显的弹窗/模态框覆盖层
   - 有明确的"错误"、"ERROR"、"失败"、"FAIL"字样
   - 有红色/黄色的警告图标或颜色
   - 注意：普通的提示文字、加载文案、频道信息、时间显示等都不是报错界面
7. **画面闪烁/撕裂**: 屏幕明显闪烁、抖动或有横向撕裂线

## 客观指标检查（必须先执行）

**在分析图像内容之前，先检查以下客观指标，它们可能暗示隐藏的缺陷：**

1. **文件大小差异**：如果 post 文件比 pre 文件小很多（通常超过 30%），而两者分辨率接近或 post 更高，这是一种强烈信号
   - 正常彩色 UI 内容：压缩率低，文件较大
   - 黑屏/深色/纯色画面：压缩率高，文件较小
   - **结论**：post 文件异常偏小 → 高度怀疑是黑屏或深色画面

2. **分辨率差异**：如果 post 分辨率与 pre 不同（宽或高少了 5 像素以上），说明画面可能发生了异常缩放或裁剪
   - **结论**：分辨率异常变化 → 可能是缺陷导致的画面异常

3. **灰暗/深色画面识别**：
   - 如果画面大部分是黑色、深灰色、且没有任何彩色 UI 元素，即使不是纯黑屏，也可能是**加载失败的黑屏前状态**
   - 这种情况应该判定为缺陷（黑屏/加载失败）

**当客观指标异常时，即使你无法100%确定，也要提高警惕、倾向于判定为 FAIL。**

## 关联性分析（重要）

**正常情况**（不是缺陷）：
- post 出现了新的菜单、新的内容页、焦点移动了、列表滚动了
- post 显示了操作反馈（如"已选中"、"加载中"等临时状态）
- 任何看起来是用户操作导致的正常界面变化

**异常情况**（是缺陷）：
- 按了遥控器，但 post 跳出了一个跟操作无关的报错界面
- 按了遥控器，但 post 变成了黑屏/花屏
- 按了遥控器，但 post 的界面完全卡死没有任何变化（排除loading状态）
- 操作后 post 出现了明显不应该出现的弹窗、错位、闪烁

**判断逻辑**：先判断 post 是不是一个正常操作结果，再判断有没有视觉缺陷。

## 绝对不是缺陷的情况（以下情况一律判定为PASS）

- 正常的加载动画/进度条/loading文字
- 频道号、频道名、收视指南等正常信息文字
- 菜单导航、按钮文字
- 时间显示、日期显示
- 底部提示文字、操作指引
- 正常的焦点高亮框（应该存在的情况）
- 焦点移动、页面切换、内容更新等正常操作结果
- 任何看起来是正常UI的文字内容

## 输出要求

请按以下 JSON 格式输出，只有在没有疑问的情况下才判定为 FAIL：
```json
{
  "pre_has_defect": true或false,
  "post_has_defect": true或false,
  "pre_defect_type": "缺陷类型，normal表示无缺陷",
  "post_defect_type": "缺陷类型，normal表示无缺陷",
  "pre_defect_description": "缺陷描述，没有缺陷则为空字符串",
  "post_defect_description": "缺陷描述，没有缺陷则为空字符串",
  "pre_defect_severity": "none/normal/moderate/severe/critical",
  "post_defect_severity": "none/normal/moderate/severe/critical",
  "overall_verdict": "PASS或FAIL",
  "confidence": 0.0到1.0之间
}
```

**判定规则**：
- 任何一张图有缺陷 → overall_verdict 为 FAIL
- 两张图都正常 → overall_verdict 为 PASS
- 如果你看到的是正常UI文字（如频道信息、提示文案等），请判定为 normal，不是缺陷

只输出 JSON，不要有其他内容。"""


def encode_image_to_base64(image_path: str) -> str:
    """将图片文件编码为 base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def call_minimax_api(payload: dict) -> dict:
    """调用 MiniMax API"""
    import urllib.request
    import urllib.error

    url = f"{MINIMAX_BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINIMAX_API_KEY}"
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        raise Exception(f"API Error {e.code}: {error_body}")
    except urllib.error.URLError as e:
        raise Exception(f"Network Error: {e.reason}")
    except Exception as e:
        raise Exception(f"Unexpected Error: {str(e)}")


def parse_model_response(response: dict) -> dict:
    """解析 API 响应"""
    try:
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("No choices in API response")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        # 去掉思考标签
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)

        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        return json.loads(content.strip())

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nContent: {content[:500]}")
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(f"Failed to extract result: {e}\n{response}")


def run_defect_check(
    pre_image_path: str,
    post_image_path: str,
    prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None
) -> dict:
    """
    执行缺陷检测

    Args:
        pre_image_path: 操作前截图路径
        post_image_path: 操作后截图路径
        prompt: 自定义提示词 (可选)
        api_key: API Key (可选)
        base_url: API 地址 (可选)
        model: 模型名称 (可选)

    Returns:
        dict: 检测结果
    """
    global MINIMAX_API_KEY, MINIMAX_BASE_URL, MINIMAX_MODEL

    if api_key:
        MINIMAX_API_KEY = api_key
    if base_url:
        MINIMAX_BASE_URL = base_url
    if model:
        MINIMAX_MODEL = model

    if not os.path.exists(pre_image_path):
        raise FileNotFoundError(f"Pre image not found: {pre_image_path}")
    if not os.path.exists(post_image_path):
        raise FileNotFoundError(f"Post image not found: {post_image_path}")

    pre_size = os.path.getsize(pre_image_path)
    post_size = os.path.getsize(post_image_path)
    pre_ratio = post_size / pre_size if pre_size > 0 else 0
    print(f"[INFO] Pre:  {pre_image_path} ({pre_size:,} bytes)")
    print(f"[INFO] Post: {post_image_path} ({post_size:,} bytes)")
    print(f"[INFO] Size ratio: post/pre = {pre_ratio:.2f}")

    # 获取分辨率
    try:
        from PIL import Image
        pre_img = Image.open(pre_image_path)
        post_img = Image.open(post_image_path)
        pre_res = pre_img.size
        post_res = post_img.size
        print(f"[INFO] Resolution: pre={pre_res}, post={post_res}")
    except ImportError:
        pre_res = None
        post_res = None

    pre_base64 = encode_image_to_base64(pre_image_path)
    post_base64 = encode_image_to_base64(post_image_path)

    actual_prompt = prompt or DEFECT_DETECTION_PROMPT

    # 构建带客观指标的消息
    meta_info = f"""
[客观指标数据]
- Pre文件大小: {pre_size:,} bytes
- Post文件大小: {post_size:,} bytes
- 文件大小比例: {pre_ratio:.2f} (post/pre)
- Pre分辨率: {pre_res}
- Post分辨率: {post_res}
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "【第一张图 pre】操作前的界面:"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{pre_base64}"}
                },
                {
                    "type": "text",
                    "text": "【第二张图 post】操作后的界面:"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{post_base64}"}
                },
                {
                    "type": "text",
                    "text": f"\n{meta_info}\n{actual_prompt}"}
            ]
        }
    ]

    payload = {
        "model": MINIMAX_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    print("[INFO] Calling MiniMax API (with 2 images)...")
    response = call_minimax_api(payload)
    result = parse_model_response(response)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="EPG Visual Defect Detection - 双图输入检测界面缺陷",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
缺陷类型:
  black_screen / white_noise / garbled_text / ui_misalign /
  focus_lost / error_popup / flicker / other / none

示例:
  python epg_defect_check.py --pre ./pre.png --post ./post.png
  python epg_defect_check.py --pre ./pre.png --post ./post.png --output result.json
        """
    )

    parser.add_argument("--pre", required=True, help="操作前的截图路径")
    parser.add_argument("--post", required=True, help="操作后的截图路径")
    parser.add_argument("--prompt", help="自定义提示词 (可选)")
    parser.add_argument("--api-key", help="MiniMax API Key (可选)")
    parser.add_argument("--base-url", help="MiniMax API 地址 (可选)")
    parser.add_argument("--model", help="模型名称 (可选)")
    parser.add_argument("--output", "-o", help="结果输出到文件 (可选)")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    args = parser.parse_args()

    try:
        print("=" * 60)
        print("EPG Visual Defect Detection")
        print("=" * 60)

        result = run_defect_check(
            pre_image_path=args.pre,
            post_image_path=args.post,
            prompt=args.prompt,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model
        )

        print("\n" + "=" * 60)
        print("检测结果:")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        verdict = result.get("overall_verdict", "UNKNOWN")
        pre_defect = result.get("pre_has_defect", False)
        post_defect = result.get("post_has_defect", False)

        print("\n" + "-" * 60)
        print(f"Pre图:  {'[FAIL] ' + result.get('pre_defect_type', '') if pre_defect else '[PASS] 正常'}")
        print(f"Post图: {'[FAIL] ' + result.get('post_defect_type', '') if post_defect else '[PASS] 正常'}")
        print(f"综合判定: [{verdict}]")
        print("-" * 60)

        if verdict == "FAIL":
            if pre_defect:
                print(f"\n[Pre缺陷] {result.get('pre_defect_description', '')}")
            if post_defect:
                print(f"[Post缺陷] {result.get('post_defect_description', '')}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n[INFO] 已保存到: {args.output}")

        sys.exit(0 if verdict == "PASS" else 1)

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(2)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()