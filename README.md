# stereo-visuals-tasks
MIT project tasks
week 2
# Stereo Vision Depth Estimation Tool User Manual
# 立体视觉深度估计工具使用说明

## Project Introduction / 项目简介
This is a stereo vision depth estimation tool based on OpenCV and PySide6, supporting both BM and SGBM stereo matching algorithms for binocular image depth map generation and analysis.
这是一个基于OpenCV和PySide6的立体视觉深度估计工具，支持BM和SGBM两种立体匹配算法，可用于双目图像深度图生成与分析。


### What's New / 更新内容:
- Optimized relative depth algorithm using percentage calculation / 优化相对深度算法，采用百分比计算
- Added English interface support / 添加英语界面支持
- Enhanced calibration file support for multiple dataset formats / 增强标定文件支持，适配多种数据集格式
  - Support for KITTI dataset calibration format / 支持KITTI数据集标定格式
  - Support for Middlebury dataset calibration format / 支持Middlebury数据集标定格式
- Improved parameter adjustment interface / 改进参数调整界面
- Fixed calibration parameter loading issues / 修复标定参数加载问题

## Features / 功能特性

### 1. Core Functions / 核心功能
- **BM Algorithm**: Fast block-based stereo matching / 基于块匹配的快速立体匹配算法
- **SGBM Algorithm**: Semi-global block matching with higher accuracy / 半全局块匹配算法，精度更高
- **Real-time Comparison**: Simultaneous comparison of both algorithms / 实时对比两种算法效果
- **Multiple Depth Calculation Methods**: / 多种深度计算方式：
  - Relative Depth (normalized) / 相对深度（归一化）
  - Absolute Depth (manual parameters) / 绝对深度（手动参数）
  - Absolute Depth (calibration file) / 绝对深度（标定文件）

### 2. Parameter Adjustment / 参数调节
- **Disparity Range**: 16-256 (multiples of 16) / 视差范围：16-256（16的倍数）
- **Block Size**: 5-255 (BM), 3-11 (SGBM) / 块大小：5-255（BM），3-11（SGBM）
- **Texture Threshold**: 0-100 / 纹理阈值：0-100
- **Uniqueness Ratio**: 0-100 / 唯一性比率：0-100
- **Speckle Filter**: Adjustable window size and range / 斑点滤波：窗口大小和范围可调
- **Preprocessing Parameters**: Filter size and truncation value / 预处理参数：滤波器大小和截断值

### 3. Data Input/Output / 数据输入输出
- **Input**: Left/right image pairs (with automatic matching) / 输入：左右图像对（支持自动匹配）
- **Output**: Color depth maps (JET colormap) / 输出：彩色深度图（JET色彩映射）
- **Save Function**: Depth maps and parameter text files / 保存功能：深度图和参数文本文件
- **Calibration Files**: Support txt/yaml/yml formats / 标定文件：支持txt/yaml/yml格式

### Software Requirements / 软件要求
- **Operating System**: Windows 10/11, Linux, macOS / Windows 10/11，Linux，macOS
- **Python Version**: 3.8+ / Python 3.8+
- **Dependencies**: / 依赖库：
  - PySide6 >= 6.5.0
  - OpenCV >= 4.5.0
  - NumPy >= 1.21.0

## Installation Steps / 安装步骤

### Method 1: Run from Source Code / 方法一：直接运行源码
```bash
# 1. Clone or download the project / 克隆或下载项目
git clone <repository-url>
cd stereo_vision_tool

# 2. Install dependencies / 安装依赖
pip install -r requirements.txt

# 3. Run the program / 运行程序
python "Depth map analysis tool.py"
```

### Method 2: Use PyInstaller .exe (Windows) / 方法二：使用PyInstaller打包后的.exe（Windows）

## Quick Start Guide / 快速入门指南

### 1. Start the Program / 启动程序
Double-click `Depth map analysis tool.exe` or run Python script / 双击`Depth map analysis tool.exe`或运行Python脚本

### 2. Load Images / 加载图像
1. Click "Browse..." to select left image / 点击"浏览..."选择左图像
2. Program automatically tries to match right image / 程序会自动尝试匹配右图像
3. If auto-match fails, manually select right image / 如果自动匹配失败，手动选择右图像

### 3. Set Parameters / 设置参数
- **Basic Parameters**: Disparity range, block size / 基本参数：视差范围、块大小
- **Optimization Parameters**: Texture threshold, uniqueness ratio / 优化参数：纹理阈值、唯一性比率
- **Depth Calculation**: Select calculation method / 深度计算：选择计算方式
  - Relative Depth: Quick preview / 相对深度：快速预览
  - Absolute Depth: Requires focal length and baseline / 绝对深度：需要焦距和基线参数

### 4. Generate Depth Map / 生成深度图
1. Select BM or SGBM algorithm tab / 选择BM或SGBM算法标签页
2. Click "Generate BM Depth Map" or "Generate SGBM Depth Map" / 点击"生成BM深度图"或"生成SGBM深度图"
3. View results in right display area / 查看右侧显示区域的结果

### 5. Comparative Analysis / 对比分析
1. Switch to "Comparison" tab / 切换到"两者对比"标签页
2. Set unified parameters / 设置统一参数
3. Click "Generate Comparison" to run both algorithms / 点击"对比生成"同时运行两种算法
4. View comparison statistics / 查看对比统计信息

### 6. Save Results / 保存结果
1. Click "Save Depth Map" for current algorithm / 点击"保存深度图"保存当前算法结果
2. Click "Save All Results" for comparison analysis / 点击"保存所有结果"保存对比分析结果
3. Files include: Depth map images and parameter text files / 文件包含：深度图图片和参数文本文件

## Parameter Details / 参数详细说明

### BM Algorithm Parameters / BM算法参数
| Parameter / 参数名称 | Default / 默认值 | Range / 范围 | Description / 说明 |
|---------------------|------------------|--------------|-------------------|
| Disparity Range / 视差范围 | 96 | 16-256 | Search range, multiples of 16 / 搜索范围，16的倍数 |
| Block Size / 块大小 | 11 | 5-255 | Matching window size, odd / 匹配窗口大小，奇数 |
| Texture Threshold / 纹理阈值 | 10 | 0-100 | Low-texture region threshold / 纹理不足区域阈值 |
| Uniqueness Ratio / 唯一性比率 | 15 | 0-100 | Matching uniqueness requirement / 匹配唯一性要求 |
| Speckle Window / 斑点窗口 | 100 | 0-1000 | Post-processing filter window / 后处理滤波窗口 |
| Speckle Range / 斑点范围 | 32 | 0-100 | Disparity change threshold / 视差变化阈值 |

### SGBM Algorithm Parameters / SGBM算法参数
| Parameter / 参数名称 | Default / 默认值 | Range / 范围 | Description / 说明 |
|---------------------|------------------|--------------|-------------------|
| Min Disparity / 最小视差 | 0 | -100-100 | Search start disparity / 搜索起始视差 |
| P1 Parameter / P1参数 | 1176 | 0-10000 | Smoothness penalty parameter 1 / 平滑惩罚参数1 |
| P2 Parameter / P2参数 | 4704 | 0-100000 | Smoothness penalty parameter 2 / 平滑惩罚参数2 |
| Algorithm Mode / 算法模式 | SGBM | SGBM/3WAY/HH | Algorithm implementation / 算法实现方式 |

### Calibration Parameters / 标定参数
| Parameter / 参数名称 | Default / 默认值 | Unit / 单位 | Description / 说明 |
|---------------------|------------------|-------------|-------------------|
| Focal Length / 焦距 | 1000 | pixels / 像素 | Camera focal length / 相机焦距 |
| Baseline / 基线 | 0.54 | meters / 米 | Distance between cameras / 相机间距离 |

## Dataset-Specific Recommendations / 数据集特定参数建议

### KITTI Dataset / KITTI数据集
**Characteristics / 特点**: Urban scenes, driving scenarios, 1242×375 resolution / 城市场景，驾驶环境，1242×375分辨率

**BM Algorithm Recommendations / BM算法建议参数**:
- Disparity Range: 96-128 / 视差范围: 96-128
- Block Size: 15-21 / 块大小: 15-21
- Texture Threshold: 5-15 / 纹理阈值: 5-15
- Uniqueness Ratio: 10-20 / 唯一性比率: 10-20

**SGBM Algorithm Recommendations / SGBM算法建议参数**:
- Disparity Range: 64-128 / 视差范围: 64-128
- Min Disparity: 0 / 最小视差: 0
- Block Size: 7-11 / 块大小: 7-11
- P1 Calculation: 8 × number_of_image_channels × block_size × block_size / P1计算: 8 × 图像通道数 × 块大小 × 块大小
  - For block_size=11, grayscale: P1 ≈ 968 / 对于块大小11，灰度图: P1 ≈ 968
  - For block_size=7, grayscale: P1 ≈ 392 / 对于块大小7，灰度图: P1 ≈ 392
- P2 Calculation: 32 × number_of_image_channels × block_size × block_size / P2计算: 32 × 图像通道数 × 块大小 × 块大小
  - For block_size=11, grayscale: P2 ≈ 3872 / 对于块大小11，灰度图: P2 ≈ 3872
  - For block_size=7, grayscale: P2 ≈ 1568 / 对于块大小7，灰度图: P2 ≈ 1568
- Mode: HH (full-scale two-pass dynamic programming) / 模式: HH (全尺度两通道动态规划)

### Middlebury Dataset / Middlebury数据集
**Characteristics / 特点**: Indoor scenes, high resolution, ground truth available / 室内场景，高分辨率，有真值数据

**BM Algorithm Recommendations / BM算法建议参数**:
- Disparity Range: 64-96 / 视差范围: 64-96
- Block Size: 9-15 / 块大小: 9-15
- Texture Threshold: 10-20 / 纹理阈值: 10-20
- Uniqueness Ratio: 15-25 / 唯一性比率: 15-25

**SGBM Algorithm Recommendations / SGBM算法建议参数**:
- **Disparity Range: 96-170** / **视差范围: 96-170**
  - For high-resolution Middlebury images (e.g., 2001×1986) / 对于高分辨率Middlebury图像（如2001×1986）
  - Typical disparity levels for indoor scenes / 室内场景的典型视差水平
- Min Disparity: 0 / 最小视差: 0
- **Block Size: 3-7** / **块大小: 3-7**
  - Smaller blocks for detailed indoor scenes / 室内场景细节较多，使用较小的块
- **P1 Calculation: 8 × number_of_image_channels × block_size × block_size** / **P1计算: 8 × 图像通道数 × 块大小 × 块大小**
  - **For block_size=7, grayscale: P1 ≈ 392** / **对于块大小7，灰度图: P1 ≈ 392**
  - For block_size=5, grayscale: P1 ≈ 200 / 对于块大小5，灰度图: P1 ≈ 200
  - For block_size=3, grayscale: P1 ≈ 72 / 对于块大小3，灰度图: P1 ≈ 72
- **P2 Calculation: 32 × number_of_image_channels × block_size × block_size** / **P2计算: 32 × 图像通道数 × 块大小 × 块大小**
  - **For block_size=7, grayscale: P2 ≈ 1568** / **对于块大小7，灰度图: P2 ≈ 1568**
  - For block_size=5, grayscale: P2 ≈ 800 / 对于块大小5，灰度图: P2 ≈ 800
  - For block_size=3, grayscale: P2 ≈ 288 / 对于块大小3，灰度图: P2 ≈ 288
- **Mode: SGBM or HH** / **模式: SGBM或HH**
  - For speed: SGBM / 速度优先: SGBM
  - For accuracy: HH / 精度优先: HH

## P1/P2 Parameter Calculation Guide / P1/P2参数计算指南

### General Principles / 通用原则
P1 and P2 parameters control the smoothness constraint in SGBM algorithm:
P1和P2参数控制SGBM算法中的平滑约束：

1. **P1** penalizes small disparity changes (1 pixel difference)
   **P1** 惩罚小的视差变化（1个像素差异）
   
2. **P2** penalizes larger disparity changes (more than 1 pixel difference)
   **P2** 惩罚较大的视差变化（超过1个像素差异）
   
3. **Rule of thumb**: P2 should be 3-5 times larger than P1
   **经验法则**: P2应该是P1的3-5倍

### Practical Recommendations / 实际建议

| Scene Type / 场景类型 | Block Size / 块大小 | P1 Range / P1范围 | P2 Range / P2范围 | P2/P1 Ratio / P2/P1比率 |
|----------------------|-------------------|------------------|------------------|------------------------|
| Indoor (Middlebury) / 室内 (Middlebury) | 3-7 | 72-392 | 288-1568 | 4.0 |
| Outdoor (KITTI) / 户外 (KITTI) | 7-11 | 392-968 | 1568-3872 | 4.0 |
| Low Texture / 低纹理 | 11-15 | 968-1800 | 3872-7200 | 4.0 |
| High Texture / 高纹理 | 5-9 | 200-648 | 800-2592 | 4.0 |

### Calculation Tool / 计算工具
Use this table to determine appropriate P1/P2 values:
使用此表确定适当的P1/P2值：

| Block Size / 块大小 | P1 (8×n×size²) / P1值 | P2 (32×n×size²) / P2值 | Recommended Scene / 推荐场景 |
|-------------------|----------------------|----------------------|----------------------------|
| 3 | 72 | 288 | Detailed Middlebury scenes / 细节丰富的Middlebury场景 |
| 5 | 200 | 800 | General Middlebury scenes / 一般Middlebury场景 |
| 7 | 392 | 1568 | **Recommended for Middlebury** / **Middlebury推荐** |
| 9 | 648 | 2592 | Outdoor scenes with medium texture / 中等纹理的户外场景 |
| 11 | 968 | 3872 | KITTI urban scenes / KITTI城市场景 |
| 13 | 1352 | 5408 | Low texture scenes / 低纹理场景 |
| 15 | 1800 | 7200 | Very low texture / 极低纹理 |

### Parameter Adjustment Suggestions / 参数调节建议
- **Indoor Scenes**: Disparity range 64-128, block size 9-15 / 室内场景：视差范围64-128，块大小9-15
- **Outdoor Scenes**: Disparity range 96-192, block size 11-21 / 室外场景：视差范围96-192，块大小11-21
- **High Texture Images**: Lower texture threshold (5-10) / 高纹理图像：降低纹理阈值（5-10）
- **Low Texture Images**: Increase texture threshold (20-50) / 低纹理图像：提高纹理阈值（20-50）

## Known Issues / 已知问题
1. **Middlebury Calibration**: Some Middlebury datasets may require manual parameter adjustment due to different calibration formats / Middlebury标定：某些Middlebury数据集可能需要手动调整参数，因为标定格式不同
2. **P1/P2 Visualization**: The calculation process for P1/P2 parameters is not currently visualized in the interface / P1/P2可视化：P1/P2参数的计算过程目前未在界面中可视化
3. **Language Switching**: English interface support is basic and may not cover all elements / 语言切换：英语界面支持基础，可能未覆盖所有元素
4.**error**: num_disparities Only multiples of 16 can be filled in; otherwise, an error will be reported./视差范围只能填16的倍数，不然报错

## Future Updates / 未来更新计划
- Add P1/P2 parameter calculation visualization / 添加P1/P2参数计算可视化
- Improve Middlebury dataset calibration support / 改进Middlebury数据集标定支持
- Add more dataset-specific presets / 添加更多数据集特定预设
- Implement batch processing functionality / 实现批处理功能

## Changelog / 更新日志

### v1.1.0 
- Optimized relative depth algorithm using percentage calculation / 优化相对深度算法，采用百分比计算
- Added English interface support / 添加英语界面支持
- Enhanced calibration file support / 增强标定文件支持
- Fixed calibration parameter loading issues / 修复标定参数加载问题
- Improved parameter adjustment interface / 改进参数调整界面

### v1.0.0 
- Initial version release / 初始版本发布
- Support for BM and SGBM algorithms / 支持BM和SGBM算法
- Graphical parameter adjustment interface / 图形化参数调节界面
- Depth map visualization display / 深度图可视化显示
- Result saving function / 结果保存功能

## License / 许可证
This project uses the MIT License. See LICENSE file for details.
本项目采用MIT许可证。详见LICENSE文件。
