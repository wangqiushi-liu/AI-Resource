import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

# 导入被测模块
from check_images import (
    encode_image,
    get_local_image_path,
    parse_model_output,
    process_row,
)


class TestGetLocalImagePath(unittest.TestCase):
    """测试 get_local_image_path 路径转换逻辑"""

    def test_windows_path(self):
        """Windows 绝对路径应提取文件名"""
        csv_path = "D:\\code\\UIAutoTest2\\runs\\run_20260604_212327\\screenshots\\000000_0003_d1d5d5c7a5c9e1ae.png"
        result = get_local_image_path(csv_path)
        self.assertEqual(result, "./images/000000_0003_d1d5d5c7a5c9e1ae.png")

    def test_unix_path(self):
        """Unix 绝对路径应提取文件名"""
        csv_path = "/Users/foo/code/screenshots/000000_0001_abc123.png"
        result = get_local_image_path(csv_path)
        self.assertEqual(result, "./images/000000_0001_abc123.png")

    def test_mixed_separators(self):
        """混合路径分隔符应统一处理"""
        csv_path = "D:/code\\UIAutoTest2/runs\\screenshots/000000_0002_def456.png"
        result = get_local_image_path(csv_path)
        self.assertEqual(result, "./images/000000_0002_def456.png")

    def test_filename_only(self):
        """仅含文件名的路径也应正常处理"""
        csv_path = "000000_0003_ghi789.png"
        result = get_local_image_path(csv_path)
        self.assertEqual(result, "./images/000000_0003_ghi789.png")


class TestEncodeImage(unittest.TestCase):
    """测试 encode_image 图片编码功能"""

    def setUp(self):
        """创建临时图片文件"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_image_path = os.path.join(self.temp_dir, "test.png")
        # 写入一个最小的有效 PNG 文件 (1x1 透明像素)
        png_bytes = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
            0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
            0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
            0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
            0x42, 0x60, 0x82
        ])
        with open(self.temp_image_path, "wb") as f:
            f.write(png_bytes)

    def tearDown(self):
        """清理临时文件"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_encode_returns_base64_string(self):
        """编码后应为有效的 base64 字符串"""
        result = encode_image(self.temp_image_path)
        self.assertIsInstance(result, str)
        # 验证 base64 格式
        import base64
        decoded = base64.b64decode(result)
        self.assertTrue(decoded.startswith(b'\x89PNG'))

    def test_encode_deterministic(self):
        """同一文件多次编码结果应一致"""
        result1 = encode_image(self.temp_image_path)
        result2 = encode_image(self.temp_image_path)
        self.assertEqual(result1, result2)

    def test_encode_nonexistent_file_raises(self):
        """文件不存在应抛出异常"""
        with self.assertRaises(FileNotFoundError):
            encode_image("/nonexistent/path/image.png")


class TestParseModelOutput(unittest.TestCase):
    """测试 parse_model_output JSON 解析逻辑"""

    def test_valid_json_with_pass(self):
        """标准 JSON 输出：PASS"""
        raw = '{"overall_verdict":"PASS","description":"焦点正确移动","defect_types":[]}'
        verdict, desc, defects = parse_model_output(raw)
        self.assertEqual(verdict, "PASS")
        self.assertEqual(desc, "焦点正确移动")
        self.assertEqual(defects, [])

    def test_valid_json_with_fail(self):
        """标准 JSON 输出：FAIL"""
        raw = '{"overall_verdict":"FAIL","description":"焦点未移动","defect_types":["A.3"]}'
        verdict, desc, defects = parse_model_output(raw)
        self.assertEqual(verdict, "FAIL")
        self.assertEqual(desc, "焦点未移动")
        self.assertEqual(defects, ["A.3"])

    def test_json_wrapped_in_code_block(self):
        """JSON 被 ```json 包裹时应正确解析"""
        raw = '```json\n{"overall_verdict":"PASS","description":"ok","defect_types":[]}\n```'
        verdict, desc, defects = parse_model_output(raw)
        self.assertEqual(verdict, "PASS")

    def test_json_wrapped_in_backticks_only(self):
        """JSON 被 ``` 包裹时应正确解析"""
        raw = '```\n{"overall_verdict":"FAIL","description":"failed","defect_types":["A.1","B.7"]}\n```'
        verdict, desc, defects = parse_model_output(raw)
        self.assertEqual(verdict, "FAIL")
        self.assertEqual(defects, ["A.1", "B.7"])

    def test_invalid_json_returns_parse_error(self):
        """非 JSON 内容应返回 Parse Error"""
        raw = "这不是 JSON 响应内容"
        verdict, desc, defects = parse_model_output(raw)
        self.assertEqual(verdict, "Parse Error")
        self.assertEqual(desc, raw)  # 原始内容应保留在 description 中

    def test_missing_overall_verdict_field(self):
        """缺少 overall_verdict 字段应返回 Unknown"""
        raw = '{"description":"有描述但无 verdict","defect_types":[]}'
        verdict, desc, defects = parse_model_output(raw)
        self.assertEqual(verdict, "Unknown")

    def test_empty_json_object(self):
        """空 JSON 对象应返回默认值"""
        raw = '{}'
        verdict, desc, defects = parse_model_output(raw)
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(desc, "")
        self.assertEqual(defects, [])

    def test_whitespace_only_json(self):
        """带前后空白的 JSON 应正常解析"""
        raw = '   \n{"overall_verdict":"PASS","description":"有空格","defect_types":[]}   \n'
        verdict, desc, defects = parse_model_output(raw)
        self.assertEqual(verdict, "PASS")

    def test_defect_types_missing_field(self):
        """缺少 defect_types 字段时应返回空数组"""
        raw = '{"overall_verdict":"FAIL","description":"无 defect_types 字段"}'
        verdict, desc, defects = parse_model_output(raw)
        self.assertEqual(verdict, "FAIL")
        self.assertEqual(defects, [])


class TestProcessRow(unittest.TestCase):
    """测试 process_row 单行处理逻辑"""

    def setUp(self):
        """准备临时图片和 CSV"""
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.images_dir)

        # 写入两张最小 PNG
        self.before_img = os.path.join(self.images_dir, "000000_0001_before.png")
        self.after_img = os.path.join(self.images_dir, "000000_0002_after.png")
        png_bytes = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
            0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
            0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
            0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
            0x42, 0x60, 0x82
        ])
        for path in [self.before_img, self.after_img]:
            with open(path, "wb") as f:
                f.write(png_bytes)

        # 临时覆盖 IMAGES_DIR
        import check_images
        self.original_images_dir = check_images.IMAGES_DIR
        check_images.IMAGES_DIR = self.images_dir

        # 构建模拟行
        self.row = pd.Series({
            "before_image": f"D:\\\\code\\\\UIAutoTest2\\\\runs\\\\screenshots\\\\000000_0001_before.png",
            "action": "DPAD_DOWN",
            "after_image": f"D:\\\\code\\\\UIAutoTest2\\\\runs\\\\screenshots\\\\000000_0002_after.png",
        })

    def tearDown(self):
        """清理"""
        import shutil
        import check_images
        shutil.rmtree(self.temp_dir)
        check_images.IMAGES_DIR = self.original_images_dir

    @patch("check_images.call_kimi_check")
    def test_process_row_returns_correct_fields(self, mock_call):
        """返回结果应包含所有必需字段"""
        mock_call.return_value = '{"overall_verdict":"PASS","description":"ok","defect_types":[]}'

        result = process_row(self.row)

        self.assertIsNotNone(result)
        self.assertIn("before_image", result)
        self.assertIn("action", result)
        self.assertIn("after_image", result)
        self.assertIn("res", result)
        self.assertIn("description", result)
        self.assertIn("defect_types", result)

    @patch("check_images.call_kimi_check")
    def test_process_row_pass_verdict(self, mock_call):
        """模型返回 PASS 时结果应为 PASS"""
        mock_call.return_value = '{"overall_verdict":"PASS","description":"焦点正确移动","defect_types":[]}'

        result = process_row(self.row)

        self.assertEqual(result["res"], "PASS")
        self.assertEqual(result["description"], "焦点正确移动")
        self.assertEqual(result["defect_types"], "")

    @patch("check_images.call_kimi_check")
    def test_process_row_fail_verdict(self, mock_call):
        """模型返回 FAIL 时结果应为 FAIL"""
        mock_call.return_value = '{"overall_verdict":"FAIL","description":"焦点未移动","defect_types":["A.3"]}'

        result = process_row(self.row)

        self.assertEqual(result["res"], "FAIL")
        self.assertEqual(result["defect_types"], "A.3")

    @patch("check_images.call_kimi_check")
    def test_process_row_parse_error(self, mock_call):
        """模型返回非 JSON 时应为 Parse Error"""
        mock_call.return_value = "这不是 JSON"

        result = process_row(self.row)

        self.assertEqual(result["res"], "Parse Error")

    @patch("check_images.call_kimi_check")
    def test_process_row_before_image_name(self, mock_call):
        """返回的 before_image 应为文件名而非完整路径"""
        mock_call.return_value = '{"overall_verdict":"PASS","description":"","defect_types":[]}'

        result = process_row(self.row)

        self.assertEqual(result["before_image"], "000000_0001_before.png")
        self.assertEqual(result["after_image"], "000000_0002_after.png")

    def test_process_row_missing_before_image(self):
        """before 图片不存在应返回 None"""
        bad_row = pd.Series({
            "before_image": "D:\\\\nonexistent\\\\path.png",
            "action": "DPAD_DOWN",
            "after_image": self.after_img,
        })

        result = process_row(bad_row)

        self.assertIsNone(result)


class TestModuleIntegration(unittest.TestCase):
    """模块级集成测试"""

    def test_all_functions_importable(self):
        """所有核心函数应可正常导入"""
        from check_images import (
            encode_image,
            get_local_image_path,
            parse_model_output,
            process_row,
            call_kimi_check,
        )
        self.assertTrue(callable(encode_image))
        self.assertTrue(callable(get_local_image_path))
        self.assertTrue(callable(parse_model_output))
        self.assertTrue(callable(process_row))
        self.assertTrue(callable(call_kimi_check))

    def test_config_values_exist(self):
        """配置常量应存在且有有效值"""
        import check_images
        self.assertTrue(hasattr(check_images, "IMAGES_DIR"))
        self.assertTrue(hasattr(check_images, "INPUT_CSV"))
        self.assertTrue(hasattr(check_images, "OUTPUT_FILE"))
        self.assertTrue(hasattr(check_images, "MAX_WORKERS"))
        self.assertIsInstance(check_images.MAX_WORKERS, int)
        self.assertGreater(check_images.MAX_WORKERS, 0)


if __name__ == "__main__":
    unittest.main()
