@echo off
chcp 65001 >nul
echo.
echo ═══════════════════════════════════════════════════════════════════
echo   老年人视频识别评测系统 - GitHub 一键上传脚本
echo ═══════════════════════════════════════════════════════════════════
echo.

:: 检查是否在正确的目录
if not exist "config.py" (
    echo [错误] 当前目录不是项目根目录！
    echo 请确保在 elderly_video_evaluation 文件夹中运行此脚本
    pause
    exit /b 1
)

echo [1/6] 配置 Git 用户信息...
git config --global user.email "pandaxiong0729@github.com"
git config --global user.name "pandaxiong0729"

echo [2/6] 初始化 Git 仓库（如果尚未初始化）...
if not exist ".git" (
    git init
)

echo [3/6] 添加远程仓库...
git remote remove origin 2>nul
git remote add origin https://github.com/pandaxiong0729/elderly_video_evaluation.git

echo [4/6] 添加所有文件到暂存区...
git add .

echo [5/6] 创建提交...
git commit -m "Initial commit: 老年人视频识别评测系统 v1.0

- 实现视频切片功能 (clips_tools.py)
- 实现批量评测功能 (run_evaluation.py)
- 实现结果分析工具 (analyze_results.py) - 支持多指标
- 实现重新评测工具 (re_evaluate.py)
- 支持 BLEU-1/2/4 和 CER 多指标评测
- 添加 config.py 集中配置管理
- 添加详细文档 (README, QUICK_REFERENCE 等)
- 支持灵活的指标列选择 (--metric 参数)
- 支持模型可插拔接入"

echo [6/6] 推送到 GitHub...
git push -u origin master

echo.
echo ═══════════════════════════════════════════════════════════════════
echo   上传完成！
echo   请访问 https://github.com/pandaxiong0729/elderly_video_evaluation 查看
echo ═══════════════════════════════════════════════════════════════════
echo.
pause
