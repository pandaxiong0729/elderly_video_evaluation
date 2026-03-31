# 更新日志

## v1.1.0 (2026-03-27)

### 新增功能
- **`--model_path` 参数**：支持通过命令行指定模型路径，无需修改代码
  ```bash
  python run_evaluation.py --model_type qwen_omni --model_path Qwen/Qwen2.5-Omni-7B
  ```

### Bug 修复
- **移除 `model_type` 的 choices 限制**：之前只能选择 `dummy`, `qwen_vl`, `intern_vl` 三种固定类型，现在支持任意自定义模型类型
- **改进错误提示**：当使用未注册的模型类型时，会显示友好的提示信息，指导用户如何添加自定义模型

### 改进
- 简化命令行参数，移除不必要的 `--mode` 参数
- 添加 `--help` 使用示例

---

## v1.0.0 (2026-03-27)

### 初始版本
- 支持 BLEU-1、CER 评测指标
- 支持文件夹模式自动匹配视频和参考文本
- 模型适配器模板（`custom_model_example.py`）
- 支持 SRT 字幕和 TXT 纯文本格式
- 自动生成 JSON、CSV、文本报告
