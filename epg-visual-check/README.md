# EPG Visual Defect Detection - IPTV 界面缺陷检测工具

使用 MiniMax 多模态大模型自动检测 IPTV 电视机 EPG 界面截图中的视觉缺陷。

## 检测范围

| 缺陷类型 | 说明 |
|---------|------|
| black_screen | 黑屏 (屏幕全黑或几乎全黑) |
| white_noise | 花屏/雪花 (条纹、色彩失真、雪花点) |
| garbled_text | 乱码 (文字无法辨认) |
| ui_misalign | UI 错位 (元素重叠、位置错误) |
| focus_lost | 焦点丢失 (高亮选框消失或位置错误) |
| error_popup | 报错 (错误提示、加载失败等) |
| flicker | 闪烁 (画面不稳定) |
| other | 其他异常 |

## 严重程度

| 等级 | 说明 |
|------|------|
| none | 无缺陷 |
| minor | 轻微 (基本不影响使用) |
| moderate | 中等 (部分功能受影响) |
| severe | 严重 (功能不可用) |
| critical | 危险 (需立即处理) |

## 安装依赖

```bash
pip install requests urllib3
```

## 配置 API Key

### 方式一: 环境变量

```bash
export MINIMAX_API_KEY="your_api_key_here"
export MINIMAX_BASE_URL="https://api.minimax.chat"
```

### 方式二: 命令行参数

```bash
python epg_defect_check.py --image ./screenshot.png --api-key your_api_key_here
```

## 使用方法

### 基本用法

```bash
python epg_defect_check.py --image /path/to/screenshot.png
```

### 输出到文件

```bash
python epg_defect_check.py --image ./screenshot.png --output result.json
```

### 自定义提示词

```bash
python epg_defect_check.py \
    --image ./screenshot.png \
    --prompt "请特别关注右下角区域是否有文字显示错误"
```

## 输出示例

```json
{
  "has_defect": false,
  "defect_type": "none",
  "defect_severity": "none",
  "defect_description": "",
  "defect_location": "",
  "recommendation": "",
  "verdict": "PASS",
  "confidence": 0.95
}
```

检测到缺陷时的输出示例:

```json
{
  "has_defect": true,
  "defect_type": "garbled_text",
  "defect_severity": "severe",
  "defect_description": "频道号显示区域出现乱码，数字和符号混杂无法辨认",
  "defect_location": "屏幕顶部居中位置",
  "recommendation": "检查 UI 渲染模块的字符编码设置",
  "verdict": "FAIL",
  "confidence": 0.92
}
```

## 判定结果

| verdict | 含义 | 返回码 |
|---------|------|--------|
| PASS | 界面正常，无缺陷 | 0 |
| FAIL | 检测到缺陷 | 1 |

## 返回码

```bash
echo $?  # 0 = PASS, 1 = FAIL
```

## 批量检测

支持批量处理多张图片:

```bash
# 假设有多张图片在 screenshots/ 目录
for img in screenshots/*.png; do
    python epg_defect_check.py --image "$img" --output "results/$(basename $img .png).json"
done
```

## 集成到自动化测试

```python
import subprocess

def check_epg_screen(image_path):
    result = subprocess.run(
        ['python', 'epg_defect_check.py', '--image', image_path, '--output', '/tmp/result.json'],
        capture_output=True
    )
    # 返回码: 0=PASS, 1=FAIL
    return result.returncode == 0
```