#!/usr/bin/env python
# coding: utf-8

import sys
import os
import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog, QGraphicsScene, QGraphicsPixmapItem, QLabel, \
    QPushButton, QWidget, QSpinBox, QDoubleSpinBox, QComboBox
from PySide6.QtCore import Qt, QFile, QIODevice, QSize
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtUiTools import QUiLoader


class StereoVisionTool:
    def __init__(self):
        # 从文件中加载UI定义
        ui_file_path = "Depth-map-analysis tool.ui"

        # 检查文件是否存在
        if not os.path.exists(ui_file_path):
            QMessageBox.critical(None, "错误", f"找不到UI文件: {ui_file_path}")
            return

        print(f"正在加载UI文件: {ui_file_path}")

        ui_file = QFile(ui_file_path)

        if not ui_file.open(QIODevice.ReadOnly):
            QMessageBox.critical(None, "错误", f"无法打开UI文件: {ui_file.errorString()}")
            return

        # 从 UI 定义中动态创建一个相应的窗口对象
        loader = QUiLoader()
        self.ui = loader.load(ui_file)
        ui_file.close()

        if not self.ui:
            QMessageBox.critical(None, "错误", "UI文件加载失败")
            return

        # 设置窗口标题
        self.ui.setWindowTitle("立体视觉深度估计工具")

        # 应用紧凑样式
        self.apply_compact_style()

        # 为所有参数添加工具提示
        self.setup_tooltips()

        # 初始化变量
        self.left_image = None
        self.right_image = None
        self.calib_file = None
        self.calib_data = None

        # 默认焦距和基线
        self.focal_length = 1000.0
        self.baseline = 0.54

        # 结果存储
        self.bm_depth = None
        self.bm_time = 0

        self.sgbm_depth = None
        self.sgbm_time = 0

        # SGBM模式映射
        self.sgbm_mode_map = {
            "SGBM": cv2.StereoSGBM_MODE_SGBM,
            "SGBM 3WAY": cv2.StereoSGBM_MODE_SGBM_3WAY,
            "HH": cv2.StereoSGBM_MODE_HH
        }

        # 连接信号和槽
        self.connect_signals()

        print("UI加载完成")
    def apply_compact_style(self):
        """应用QSS样式表，使框和按钮更明显"""
        qss = """
        /* 主窗口样式 */
        QMainWindow {
            background-color: #f5f5f5;
        }

        /* 分组框样式 - 更加明显 */
        QGroupBox {
            font-weight: bold;
            border: 2px solid #3498db;
            border-radius: 6px;
            margin-top: 6px;
            padding-top: 6px;
            background-color: #ffffff;
            color: #2c3e50;
            font-size: 12px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px ;
            background-color: #3498db;
            color: white;
            border-radius: 2px;
            font-size: 11px;
        }

        /* 主要操作按钮 - 更明显 */
        QPushButton {
            font-weight: bold;
            border: 1px solid #2c3e50;
            border-radius: 4px;
            padding: 4px 8px;
            background-color: #3498db;
            color: white;
            font-size: 11px;
            min-width: 60px;
            max-height: 25px;
        }

        QPushButton:hover {
            background-color: #2980b9;
            border-color: #3498db;
        }

        QPushButton:pressed {
            background-color: #21618c;
            border-color: #1b4f72;
        }

        /* 生成按钮特殊样式 */
        QPushButton[name*="Generate"] {
            background-color: #27ae60;
            border-color: #229954;
            font-size: 12px;
            font-weight: bold;
            padding: 6px 12px;
            min-width: 80px;
        }

        QPushButton[name*="Generate"]:hover {
            background-color: #229954;
            border-color: #1e8449;
        }

        QPushButton[name*="Generate"]:pressed {
            background-color: #1e8449;
            border-color: #186a3b;
        }

        /* 保存按钮特殊样式 */
        QPushButton[name*="Save"] {
            background-color: #e74c3c;
            border-color: #c0392b;
            font-size: 11px;
            padding: 4px 8px;
            min-width: 50px;
        }

        QPushButton[name*="Save"]:hover {
            background-color: #c0392b;
            border-color: #a93226;
        }

        QPushButton[name*="Save"]:pressed {
            background-color: #a93226;
            border-color: #922b21;
        }

        /* 加载/浏览按钮 */
        QPushButton[name*="Load"] {
            background-color: #9b59b6;
            border-color: #8e44ad;
        }

        QPushButton[name*="Load"]:hover {
            background-color: #8e44ad;
            border-color: #7d3c98;
        }

        /* 预览按钮 */
        QPushButton[name*="Show"] {
            background-color: #f39c12;
            border-color: #e67e22;
        }

        QPushButton[name*="Show"]:hover {
            background-color: #e67e22;
            border-color: #d35400;
        }

        /* 选项卡样式 */
        QTabWidget::pane {
            border: 2px solid #3498db;
            border-radius: 3px;
            background-color: #ffffff;
        }

        QTabBar::tab {
            background-color: #ecf0f1;
            border: 1px solid #bdc3c7;
            padding: 4px 8px;
            margin-right: 1px;
            border-top-left-radius: 3px;
            border-top-right-radius: 3px;
            font-weight: bold;
            color: #7f8c8d;
            font-size: 11px;
        }

        QTabBar::tab:selected {
            background-color: #3498db;
            color: white;
            border-bottom-color: #3498db;
        }

        QTabBar::tab:hover:!selected {
            background-color: #d5dbdb;
        }

        /* 输入框样式 */
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            border: 1px solid #bdc3c7;
            border-radius: 3px;
            padding: 3px;
            background-color: white;
            selection-background-color: #3498db;
            font-size: 11px;
            min-height: 20px;
            max-height: 22px;
            color: #2c3e50;
        }

        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #3498db;
        }

        /* 标签样式 */
        QLabel {
            font-weight: bold;
            color: #2c3e50;
            font-size: 11px;
        }

        QLabel[name*="Time"],
        QLabel[name*="Info"],
        QLabel[name*="ImageInfo"] {
            font-weight: bold;
            color: #e74c3c;
            font-size: 10px;
            padding: 1px 4px;
            background-color: #f9ebea;
            border-radius: 2px;
        }

        /* 图形视图样式 */
        QGraphicsView {
            border: 2px solid #7f8c8d;
            border-radius: 4px;
            background-color: #ecf0f1;
        }

        /* 文本编辑框样式 */
        QTextEdit {
            border: 1px solid #bdc3c7;
            border-radius: 3px;
            background-color: white;
            font-family: "Courier New", monospace;
            font-size: 10px;
        }

        /* 预览标签 - 白色背景，深色文字 */
        QLabel#label_Preview {
            border: 2px solid #3498db;
            border-radius: 4px;
            background-color: white;
            color: #2c3e50;
            font-weight: bold;
            padding: 2px;
            font-size: 11px;
        }

        /* 侧边栏分组框特殊样式 */
        #groupBox_ImageInput,
        #groupBox_Calibration,
        #groupBox_Preview {
            border: 2px solid #9b59b6;
        }

        #groupBox_ImageInput::title,
        #groupBox_Calibration::title,
        #groupBox_Preview::title {
            background-color: #9b59b6;
        }

        /* 对比统计分组框特殊样式 */
        #groupBox_Compare_Stats {
            border: 2px solid #f39c12;
        }

        #groupBox_Compare_Stats::title {
            background-color: #f39c12;
        }

        /* BM相关分组框特殊样式 */
        #groupBox_BM_Params,
        #groupBox_BM_Result,
        #groupBox_Compare_BM {
            border: 2px solid #3498db;
        }

        #groupBox_BM_Params::title,
        #groupBox_BM_Result::title,
        #groupBox_Compare_BM::title {
            background-color: #3498db;
        }

        /* SGBM相关分组框特殊样式 */
        #groupBox_SGBM_Params,
        #groupBox_SGBM_Result,
        #groupBox_Compare_SGBM {
            border: 2px solid #27ae60;
        }

        #groupBox_SGBM_Params::title,
        #groupBox_SGBM_Result::title,
        #groupBox_Compare_SGBM::title {
            background-color: #27ae60;
        }

        /* 对比参数分组框 */
        #groupBox_Compare_Params {
            border: 2px solid #f39c12;
        }

        #groupBox_Compare_Params::title {
            background-color: #f39c12;
        }
        """

        # 应用QSS样式
        self.ui.setStyleSheet(qss)

        # 为预览标签设置更特殊的样式 - 白色背景，蓝色边框
        preview_label = self.ui.findChild(QLabel, "label_Preview")
        if preview_label:
            preview_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #3498db;
                    border-radius: 4px;
                    background-color: white;
                    color: #2c3e50;
                    font-weight: bold;
                    padding: 2px;
                    font-size: 11px;
                }
            """)
    # def apply_compact_style(self):
    #     """应用QSS样式表，使框和按钮更明显"""
    #     qss = """
    #     /* 主窗口样式 */
    #     QMainWindow {
    #         background-color: #f5f5f5;
    #     }
    #
    #     /* 分组框样式 - 更加明显 */
    #     QGroupBox {
    #         font-weight: bold;
    #         border: 2px solid #3498db;
    #         border-radius: 6px;
    #         margin-top: 6px;
    #         padding-top: 6px;
    #         background-color: #ffffff;
    #         color: #2c3e50;
    #         font-size: 12px;
    #     }
    #
    #     QGroupBox::title {
    #         subcontrol-origin: margin;
    #         left: 8px;
    #         padding: 0 4px ;
    #         background-color: #3498db;
    #         color: white;
    #         border-radius: 2px;
    #         font-size: 11px;
    #     }
    #
    #     /* 主要操作按钮 - 更明显 */
    #     QPushButton {
    #         font-weight: bold;
    #         border: 1px solid #2c3e50;
    #         border-radius: 4px;
    #         padding: 4px 8px;
    #         background-color: #3498db;
    #         color: white;
    #         font-size: 11px;
    #         min-width: 60px;
    #         max-height: 25px;
    #     }
    #
    #     QPushButton:hover {
    #         background-color: #2980b9;
    #         border-color: #3498db;
    #     }
    #
    #     QPushButton:pressed {
    #         background-color: #21618c;
    #         border-color: #1b4f72;
    #     }
    #
    #     /* 生成按钮特殊样式 */
    #     QPushButton[name*="Generate"] {
    #         background-color: #27ae60;
    #         border-color: #229954;
    #         font-size: 12px;
    #         font-weight: bold;
    #         padding: 6px 12px;
    #         min-width: 80px;
    #     }
    #
    #     QPushButton[name*="Generate"]:hover {
    #         background-color: #229954;
    #         border-color: #1e8449;
    #     }
    #
    #     QPushButton[name*="Generate"]:pressed {
    #         background-color: #1e8449;
    #         border-color: #186a3b;
    #     }
    #
    #     /* 保存按钮特殊样式 */
    #     QPushButton[name*="Save"] {
    #         background-color: #e74c3c;
    #         border-color: #c0392b;
    #         font-size: 11px;
    #         padding: 4px 8px;
    #         min-width: 50px;
    #     }
    #
    #     QPushButton[name*="Save"]:hover {
    #         background-color: #c0392b;
    #         border-color: #a93226;
    #     }
    #
    #     QPushButton[name*="Save"]:pressed {
    #         background-color: #a93226;
    #         border-color: #922b21;
    #     }
    #
    #     /* 加载/浏览按钮 */
    #     QPushButton[name*="Load"] {
    #         background-color: #9b59b6;
    #         border-color: #8e44ad;
    #     }
    #
    #     QPushButton[name*="Load"]:hover {
    #         background-color: #8e44ad;
    #         border-color: #7d3c98;
    #     }
    #
    #     /* 预览按钮 */
    #     QPushButton[name*="Show"] {
    #         background-color: #f39c12;
    #         border-color: #e67e22;
    #     }
    #
    #     QPushButton[name*="Show"]:hover {
    #         background-color: #e67e22;
    #         border-color: #d35400;
    #     }
    #
    #     /* 选项卡样式 */
    #     QTabWidget::pane {
    #         border: 2px solid #3498db;
    #         border-radius: 3px;
    #         background-color: #ffffff;
    #     }
    #
    #     QTabBar::tab {
    #         background-color: #ecf0f1;
    #         border: 1px solid #bdc3c7;
    #         padding: 4px 8px;
    #         margin-right: 1px;
    #         border-top-left-radius: 3px;
    #         border-top-right-radius: 3px;
    #         font-weight: bold;
    #         color: #7f8c8d;
    #         font-size: 11px;
    #     }
    #
    #     QTabBar::tab:selected {
    #         background-color: #3498db;
    #         color: white;
    #         border-bottom-color: #3498db;
    #     }
    #
    #     QTabBar::tab:hover:!selected {
    #         background-color: #d5dbdb;
    #     }
    #
    #     /* 输入框样式 */
    #     QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    #         border: 1px solid #bdc3c7;
    #         border-radius: 3px;
    #         padding: 3px;
    #         background-color: white;
    #         selection-background-color: #3498db;
    #         font-size: 11px;
    #         min-height: 20px;
    #         max-height: 22px;
    #         color: #2c3e50;
    #     }
    #
    #     QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    #         border-color: #3498db;
    #     }
    #
    #     /* 标签样式 */
    #     QLabel {
    #         font-weight: bold;
    #         color: #2c3e50;
    #         font-size: 11px;
    #     }
    #
    #     QLabel[name*="Time"],
    #     QLabel[name*="Info"],
    #     QLabel[name*="ImageInfo"] {
    #         font-weight: bold;
    #         color: #e74c3c;
    #         font-size: 10px;
    #         padding: 1px 4px;
    #         background-color: #f9ebea;
    #         border-radius: 2px;
    #     }
    #
    #     /* 图形视图样式 */
    #     QGraphicsView {
    #         border: 2px solid #7f8c8d;
    #         border-radius: 4px;
    #         background-color: #ecf0f1;
    #     }
    #
    #     /* 文本编辑框样式 */
    #     QTextEdit {
    #         border: 1px solid #bdc3c7;
    #         border-radius: 3px;
    #         background-color: white;
    #         font-family: "Courier New", monospace;
    #         font-size: 10px;
    #     }
    #
    #     /* 预览标签 */
    #     QLabel#label_Preview {
    #         border: 1px solid #34495e;
    #         border-radius: 4px;
    #         background-color: #2c3e50;
    #         color: #ecf0f1;
    #         font-weight: bold;
    #         padding: 2px;
    #         font-size: 10px;
    #     }
    #
    #     /* 侧边栏分组框特殊样式 */
    #     #groupBox_ImageInput,
    #     #groupBox_Calibration,
    #     #groupBox_Preview {
    #         border: 2px solid #9b59b6;
    #     }
    #
    #     #groupBox_ImageInput::title,
    #     #groupBox_Calibration::title,
    #     #groupBox_Preview::title {
    #         background-color: #9b59b6;
    #     }
    #
    #     /* 对比统计分组框特殊样式 */
    #     #groupBox_Compare_Stats {
    #         border: 2px solid #f39c12;
    #     }
    #
    #     #groupBox_Compare_Stats::title {
    #         background-color: #f39c12;
    #     }
    #
    #     /* BM相关分组框特殊样式 */
    #     #groupBox_BM_Params,
    #     #groupBox_BM_Result,
    #     #groupBox_Compare_BM {
    #         border: 2px solid #3498db;
    #     }
    #
    #     #groupBox_BM_Params::title,
    #     #groupBox_BM_Result::title,
    #     #groupBox_Compare_BM::title {
    #         background-color: #3498db;
    #     }
    #
    #     /* SGBM相关分组框特殊样式 */
    #     #groupBox_SGBM_Params,
    #     #groupBox_SGBM_Result,
    #     #groupBox_Compare_SGBM {
    #         border: 2px solid #27ae60;
    #     }
    #
    #     #groupBox_SGBM_Params::title,
    #     #groupBox_SGBM_Result::title,
    #     #groupBox_Compare_SGBM::title {
    #         background-color: #27ae60;
    #     }
    #
    #     /* 对比参数分组框 */
    #     #groupBox_Compare_Params {
    #         border: 2px solid #f39c12;
    #     }
    #
    #     #groupBox_Compare_Params::title {
    #         background-color: #f39c12;
    #     }
    #     """
    #
    #     # 应用QSS样式
    #     self.ui.setStyleSheet(qss)
    #
    #     # 为预览标签设置更特殊的样式（因为它是动态创建的）
    #     preview_label = self.ui.findChild(QLabel, "label_Preview")
    #     if preview_label:
    #         preview_label.setStyleSheet("""
    #             QLabel {
    #                 border: 1px solid #34495e;
    #                 border-radius: 4px;
    #                 background-color: #2c3e50;
    #                 color: #ecf0f1;
    #                 font-weight: bold;
    #                 padding: 2px;
    #                 font-size: 10px;
    #             }
    #         """)

    def setup_tooltips(self):
        """为所有参数控件设置工具提示"""

        # BM算法参数
        self.ui.spinBox_BM_numDisparities.setToolTip(
            "视差范围: 必须能被16整除的整数\n"
            "定义了匹配搜索的范围，值越大能检测的深度范围越大，但计算时间也越长\n"
            "通常设置为16的倍数，如16、32、48等"
        )

        self.ui.spinBox_BM_blockSize.setToolTip(
            "块大小: 用于匹配的窗口大小，必须是奇数\n"
            "值越大，平滑效果越好但细节丢失越多\n"
            "值越小，细节保留越好但噪声越多\n"
            "建议范围: 5-21"
        )

        self.ui.spinBox_BM_textureThreshold.setToolTip(
            "纹理阈值: 用于检测纹理不足区域\n"
            "当窗口内纹理不足时，不计算该点的视差\n"
            "值越大，越容易判断为纹理不足区域\n"
            "通常设置为5-20"
        )

        self.ui.spinBox_BM_uniquenessRatio.setToolTip(
            "唯一性比率: 衡量最佳匹配的唯一性\n"
            "值越大，匹配要求越严格，减少误匹配\n"
            "通常设置为5-15"
        )

        self.ui.spinBox_BM_speckleWindowSize.setToolTip(
            "斑点窗口: 用于后处理的平滑窗口大小\n"
            "值越大，平滑效果越强\n"
            "设置为0禁用斑点滤波器"
        )

        self.ui.spinBox_BM_speckleRange.setToolTip(
            "斑点范围: 斑点滤波器的视差变化阈值\n"
            "值越大，允许的视差变化越大\n"
            "通常设置为16或32"
        )

        self.ui.spinBox_BM_disp12MaxDiff.setToolTip(
            "视差差异: 左右一致性检查的最大允许差异\n"
            "用于检测错误匹配\n"
            "值越大，容忍度越高\n"
            "-1表示禁用左右一致性检查"
        )

        self.ui.spinBox_BM_preFilterSize.setToolTip(
            "预处理大小: 预处理滤波器的窗口大小\n"
            "必须是奇数，通常在5-255之间\n"
            "用于图像的预滤波"
        )

        self.ui.spinBox_BM_preFilterCap.setToolTip(
            "预处理截断: 预处理滤波器的截断值\n"
            "通常在1-31之间\n"
            "值越大，对比度增强越强"
        )

        self.ui.comboBox_BM_DepthType.setToolTip(
            "深度计算方式:\n"
            "1. 相对深度（归一化）- 仅显示相对深度信息，无实际物理单位\n"
            "2. 绝对深度（手动参数）- 使用手动输入的焦距和基线计算实际深度\n"
            "3. 绝对深度（标定文件）- 使用标定文件中的参数计算实际深度"
        )

        # SGBM算法参数
        self.ui.spinBox_SGBM_minDisparity.setToolTip(
            "最小视差: 搜索的起始视差值\n"
            "可以为负数，表示可以搜索负视差\n"
            "通常设置为0"
        )

        self.ui.spinBox_SGBM_numDisparities.setToolTip(
            "视差范围: 必须能被16整除的整数\n"
            "定义了匹配搜索的范围\n"
            "通常设置为16的倍数"
        )

        self.ui.spinBox_SGBM_blockSize.setToolTip(
            "块大小: 用于匹配的窗口大小，必须是奇数\n"
            "SGBM中建议使用较小的值，如3、5、7"
        )

        self.ui.spinBox_SGBM_P1.setToolTip(
            "P1参数: 控制视差平滑度的第一个惩罚参数\n"
            "P1是相邻像素间视差变化为1时的惩罚\n"
            "通常设置为α * 通道数 * 块大小²\n"
            "通常α = 4, 8, 16"
        )

        self.ui.spinBox_SGBM_P2.setToolTip(
            "P2参数: 控制视差平滑度的第二个惩罚参数\n"
            "P2是相邻像素间视差变化大于1时的惩罚\n"
            "通常设置为β * 通道数 * 块大小²\n"
            "通常 β = 16, 32, 64, 96\n"
            "P2应大于P1"
        )

        self.ui.spinBox_SGBM_disp12MaxDiff.setToolTip(
            "视差差异: 左右一致性检查的最大允许差异\n"
            "用于检测错误匹配\n"
            "-1表示禁用检查"
        )

        self.ui.spinBox_SGBM_uniquenessRatio.setToolTip(
            "唯一性比率: 衡量最佳匹配的唯一性\n"
            "通常设置为5-15"
        )

        self.ui.spinBox_SGBM_speckleWindowSize.setToolTip(
            "斑点窗口: 用于后处理的平滑窗口大小\n"
            "设置为0禁用斑点滤波器"
        )

        self.ui.spinBox_SGBM_speckleRange.setToolTip(
            "斑点范围: 斑点滤波器的视差变化阈值\n"
            "通常设置为16或32"
        )

        self.ui.spinBox_SGBM_preFilterCap.setToolTip(
            "预处理截断: 预处理滤波器的截断值\n"
            "通常在1-63之间"
        )

        self.ui.comboBox_SGBM_mode.setToolTip(
            "算法模式:\n"
            "SGBM - 标准SGBM算法\n"
            "SGBM 3WAY - 使用3方向动态规划，速度更快\n"
            "HH - 使用更精确但更慢的全动态规划"
        )

        self.ui.comboBox_SGBM_DepthType.setToolTip(
            "深度计算方式:\n"
            "1. 相对深度（归一化）- 仅显示相对深度信息\n"
            "2. 绝对深度（手动参数）- 使用手动参数计算实际深度\n"
            "3. 绝对深度（标定文件）- 使用标定文件参数计算实际深度"
        )

        # 对比参数
        self.ui.spinBox_Compare_numDisparities.setToolTip(
            "视差范围: 统一应用于BM和SGBM的视差范围"
        )

        self.ui.spinBox_Compare_blockSize.setToolTip(
            "块大小: 统一应用于BM和SGBM的块大小"
        )

        self.ui.spinBox_Compare_uniquenessRatio.setToolTip(
            "唯一性比率: 统一应用于BM和SGBM的唯一性比率"
        )

        self.ui.spinBox_Compare_speckleWindowSize.setToolTip(
            "斑点窗口: 统一应用于BM和SGBM的斑点窗口大小"
        )

        self.ui.comboBox_Compare_DepthType.setToolTip(
            "深度计算方式: 统一应用于对比分析的深度计算方式"
        )

        # 标定参数
        self.ui.doubleSpinBox_FocalLength.setToolTip(
            "焦距: 相机的焦距，单位为像素\n"
            "用于从视差计算绝对深度\n"
            "典型值: 500-2000像素"
        )

        self.ui.doubleSpinBox_Baseline.setToolTip(
            "基线: 两个相机之间的距离，单位为米\n"
            "用于从视差计算绝对深度\n"
            "典型值: 0.1-1.0米"
        )

        # 图像输入
        self.ui.lineEdit_LeftImage.setToolTip("左相机的图像文件路径")
        self.ui.lineEdit_RightImage.setToolTip("右相机的图像文件路径")
        self.ui.lineEdit_CalibFile.setToolTip("相机标定文件路径（可选）\n支持.txt, .yaml, .yml格式")

        # 按钮
        self.ui.pushButton_BM_Generate.setToolTip("使用BM算法生成深度图")
        self.ui.pushButton_SGBM_Generate.setToolTip("使用SGBM算法生成深度图")
        self.ui.pushButton_Compare_Generate.setToolTip("使用相同参数对比两种算法")
        self.ui.pushButton_BM_Save.setToolTip("保存BM算法生成的深度图")
        self.ui.pushButton_SGBM_Save.setToolTip("保存SGBM算法生成的深度图")
        self.ui.pushButton_Compare_SaveAll.setToolTip("保存所有对比结果")
        self.ui.pushButton_ShowLeft.setToolTip("在预览区域显示左图像")
        self.ui.pushButton_ShowRight.setToolTip("在预览区域显示右图像")

        # 深度图显示区域
        self.ui.graphicsView_BM.setToolTip("BM算法生成的深度图显示区域")
        self.ui.graphicsView_SGBM.setToolTip("SGBM算法生成的深度图显示区域")
        self.ui.graphicsView_Compare_BM.setToolTip("对比分析中的BM深度图")
        self.ui.graphicsView_Compare_SGBM.setToolTip("对比分析中的SGBM深度图")

        print("工具提示设置完成")

    def connect_signals(self):
        """连接所有信号和槽"""
        # 图像加载按钮
        self.ui.pushButton_LoadLeft.clicked.connect(self.load_left_image)
        self.ui.pushButton_LoadRight.clicked.connect(self.load_right_image)
        self.ui.pushButton_LoadCalib.clicked.connect(self.load_calib_file)

        # 预览按钮
        self.ui.pushButton_ShowLeft.clicked.connect(lambda: self.show_image_preview('left'))
        self.ui.pushButton_ShowRight.clicked.connect(lambda: self.show_image_preview('right'))

        # 标定参数更新
        self.ui.doubleSpinBox_FocalLength.valueChanged.connect(self.update_focal_length)
        self.ui.doubleSpinBox_Baseline.valueChanged.connect(self.update_baseline)

        # BM选项卡
        self.ui.pushButton_BM_Generate.clicked.connect(self.generate_bm_depth)
        self.ui.pushButton_BM_Save.clicked.connect(lambda: self.save_result('bm'))

        # SGBM选项卡
        self.ui.pushButton_SGBM_Generate.clicked.connect(self.generate_sgbm_depth)
        self.ui.pushButton_SGBM_Save.clicked.connect(lambda: self.save_result('sgbm'))

        # 对比选项卡
        self.ui.pushButton_Compare_Generate.clicked.connect(self.compare_algorithms)
        self.ui.pushButton_Compare_SaveAll.clicked.connect(self.save_all_results)

        print("所有信号连接完成")

    def update_focal_length(self, value):
        """更新焦距"""
        self.focal_length = value
        print(f"焦距更新为: {value}")

    def update_baseline(self, value):
        """更新基线"""
        self.baseline = value
        print(f"基线更新为: {value}")

    def parse_calib_file(self, file_path):
        """解析标定文件"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                print(f"标定文件内容:\n{content[:500]}...")  # 只显示前500字符

            self.calib_data = {}
            # 简单的解析逻辑
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # 尝试不同的分隔符
                if '=' in line:
                    key, value = line.split('=', 1)
                elif ':' in line:
                    key, value = line.split(':', 1)
                else:
                    continue

                key = key.strip()
                value = value.strip()
                self.calib_data[key] = value

                # 提取焦距和基线
                if 'focal' in key.lower():
                    try:
                        self.focal_length = float(value)
                        self.ui.doubleSpinBox_FocalLength.setValue(self.focal_length)
                        print(f"找到焦距: {self.focal_length}")
                    except:
                        pass
                elif 'baseline' in key.lower():
                    try:
                        self.baseline = float(value)
                        self.ui.doubleSpinBox_Baseline.setValue(self.baseline)
                        print(f"找到基线: {self.baseline}")
                    except:
                        pass

            print("标定文件解析成功")
            QMessageBox.information(self.ui, "成功",
                                    f"标定文件解析成功\n焦距: {self.focal_length}\n基线: {self.baseline}")

        except Exception as e:
            print(f"标定文件解析错误: {e}")
            QMessageBox.warning(self.ui, "警告", f"标定文件解析错误: {str(e)}")

    def load_left_image(self):
        """加载左图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.ui, "选择左图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)")

        if file_path:
            self.ui.lineEdit_LeftImage.setText(file_path)
            self.left_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.left_image is not None:
                print(f"左图像加载成功: {Path(file_path).name}, 尺寸: {self.left_image.shape}")
                self.show_image_preview('left')
                self.auto_load_right_image(file_path)
                self.update_image_info()
            else:
                QMessageBox.warning(self.ui, "错误", "无法加载左图像")

    def load_right_image(self):
        """加载右图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.ui, "选择右图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)")

        if file_path:
            self.ui.lineEdit_RightImage.setText(file_path)
            self.right_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.right_image is not None:
                print(f"右图像加载成功: {Path(file_path).name}, 尺寸: {self.right_image.shape}")
                self.show_image_preview('right')
                self.update_image_info()
            else:
                QMessageBox.warning(self.ui, "错误", "无法加载右图像")

    def load_calib_file(self):
        """加载标定文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.ui, "选择标定文件", "",
            "标定文件 (*.txt *.yaml *.yml);;所有文件 (*)")

        if file_path:
            self.ui.lineEdit_CalibFile.setText(file_path)
            self.calib_file = file_path
            self.parse_calib_file(file_path)

    def show_image_preview(self, image_type):
        """显示图像预览"""
        if image_type == 'left' and self.left_image is not None:
            self.display_image(self.left_image, self.ui.label_Preview)
        elif image_type == 'right' and self.right_image is not None:
            self.display_image(self.right_image, self.ui.label_Preview)
        else:
            self.ui.label_Preview.setText("无图像")
            self.ui.label_Preview.setPixmap(QPixmap())

    def update_image_info(self):
        """更新图像信息显示"""
        if self.left_image is not None:
            h, w = self.left_image.shape
            image_info = f"图像尺寸: {w}×{h}"
            # 更新所有选项卡的图像信息
            self.ui.label_BM_ImageInfo.setText(image_info)
            self.ui.label_SGBM_ImageInfo.setText(image_info)

    def display_image(self, image, label_widget):
        """在QLabel中显示图像 - 紧凑版本"""
        if image is None:
            return

        # 将OpenCV图像转换为QImage
        if len(image.shape) == 2:  # 灰度图
            h, w = image.shape
            bytes_per_line = w
            qimage = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:  # 彩色图
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qimage = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 获取标签尺寸，调整图像大小
        label_size = label_widget.size()
        if label_size.width() > 0 and label_size.height() > 0:
            # 计算适合标签的尺寸
            target_width = label_size.width() - 4  # 留出边框
            target_height = label_size.height() - 4

            # 保持宽高比
            aspect_ratio = w / h

            if target_width / aspect_ratio <= target_height:
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                new_height = target_height
                new_width = int(target_height * aspect_ratio)

            # 创建缩略图
            qimage_scaled = qimage.scaled(
                new_width, new_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            label_widget.setPixmap(QPixmap.fromImage(qimage_scaled))
            label_widget.setScaledContents(False)  # 不要拉伸图像

    def check_images_loaded(self):
        """检查图像是否已加载"""
        if self.left_image is None:
            QMessageBox.warning(self.ui, "错误", "请先加载左图像")
            return False

        if self.right_image is None:
            QMessageBox.warning(self.ui, "错误", "请先加载右图像")
            return False

        return True

    def validate_images_size(self):
        """验证图像尺寸是否一致"""
        if self.left_image.shape != self.right_image.shape:
            error_msg = f"图像尺寸不一致: 左图{self.left_image.shape}, 右图{self.right_image.shape}"
            QMessageBox.warning(self.ui, "错误", error_msg)
            return False
        return True

    def get_depth_calculation_type(self, combo_box):
        """获取深度计算类型"""
        index = combo_box.currentIndex()
        if index == 0:
            return False, "相对深度（归一化）", "归一化值"
        elif index == 1:
            return True, "绝对深度（手动参数）", "米"
        elif index == 2:
            return True, "绝对深度（标定文件）", "米"
        else:
            return False, "相对深度（归一化）", "归一化值"

    def compute_depth_from_disparity(self, disparity, use_calib):
        """从视差图计算深度图"""
        if disparity is None:
            return None

        # 避免除零错误
        disparity_filtered = disparity.copy().astype(np.float32)
        disparity_filtered[disparity_filtered <= 0] = 0.1

        if use_calib:
            # 使用标定参数计算绝对深度
            depth_map = (self.focal_length * self.baseline) / disparity_filtered

            # 限制最大深度
            max_depth = 100.0  # 米
            depth_map[depth_map > max_depth] = max_depth
            depth_map[depth_map < 0] = 0

            print(f"计算绝对深度，焦距={self.focal_length}, 基线={self.baseline}")
        else:
            # 相对深度（归一化）
            relative_depth = 1.0 / disparity_filtered

            # 统计归一化
            valid_values = relative_depth[~np.isinf(relative_depth)]
            if len(valid_values) > 0:
                p95 = np.percentile(valid_values, 95)
                relative_depth = np.clip(relative_depth, 0, p95)
                relative_depth = relative_depth / p95

            # 归一化到0-255范围用于显示
            depth_map = cv2.normalize(relative_depth, None, 0, 255, cv2.NORM_MINMAX)
            print("计算相对深度")

        return depth_map

    def generate_bm_depth(self):
        """生成BM深度图"""
        print("开始生成BM深度图")

        if not self.check_images_loaded() or not self.validate_images_size():
            return

        # 获取深度计算方式
        use_calib, depth_type, depth_unit = self.get_depth_calculation_type(
            self.ui.comboBox_BM_DepthType)

        # 获取BM参数
        num_disparities = self.ui.spinBox_BM_numDisparities.value()
        block_size = self.ui.spinBox_BM_blockSize.value()
        texture_threshold = self.ui.spinBox_BM_textureThreshold.value()
        uniqueness_ratio = self.ui.spinBox_BM_uniquenessRatio.value()
        speckle_window_size = self.ui.spinBox_BM_speckleWindowSize.value()
        speckle_range = self.ui.spinBox_BM_speckleRange.value()
        disp12_max_diff = self.ui.spinBox_BM_disp12MaxDiff.value()
        pre_filter_size = self.ui.spinBox_BM_preFilterSize.value()
        pre_filter_cap = self.ui.spinBox_BM_preFilterCap.value()

        print(f"BM参数: num_disparities={num_disparities}, block_size={block_size}")
        print(f"BM参数: texture_threshold={texture_threshold}, uniqueness_ratio={uniqueness_ratio}")
        print(f"深度计算方式: {depth_type}")

        # 开始计时
        start_time = time.time()

        try:
            # 确保块大小是奇数
            if block_size % 2 == 0:
                block_size += 1
                self.ui.spinBox_BM_blockSize.setValue(block_size)

            # 创建BM对象
            stereo = cv2.StereoBM_create(
                numDisparities=num_disparities,
                blockSize=block_size
            )

            # 设置其他参数
            stereo.setTextureThreshold(texture_threshold)
            stereo.setUniquenessRatio(uniqueness_ratio)
            stereo.setSpeckleWindowSize(speckle_window_size)
            stereo.setSpeckleRange(speckle_range)
            stereo.setDisp12MaxDiff(disp12_max_diff)
            stereo.setPreFilterSize(pre_filter_size)
            stereo.setPreFilterCap(pre_filter_cap)

            # 计算视差
            print("正在计算视差...")
            disparity = stereo.compute(self.left_image, self.right_image)
            print("视差计算完成")

            # 计算深度图
            self.bm_depth = self.compute_depth_from_disparity(disparity, use_calib)

            self.bm_time = time.time() - start_time

            # 显示结果
            self.display_depth_result(self.bm_depth, self.ui.graphicsView_BM)
            self.ui.label_BM_Time.setText(f"计算时间: {self.bm_time:.3f}秒")
            self.ui.label_BM_DepthInfo.setText(f"深度类型: {depth_type} ({depth_unit})")

            # QMessageBox.information(self.ui, "成功",
            #                         f"BM深度图生成完成\n耗时: {self.bm_time:.3f}秒\n深度类型: {depth_type}")

        except Exception as e:
            error_msg = f"BM计算失败: {str(e)}"
            print(error_msg)
            QMessageBox.critical(self.ui, "错误", error_msg)

    def generate_sgbm_depth(self):
        """生成SGBM深度图"""
        print("开始生成SGBM深度图")

        if not self.check_images_loaded() or not self.validate_images_size():
            return

        # 获取深度计算方式
        use_calib, depth_type, depth_unit = self.get_depth_calculation_type(
            self.ui.comboBox_SGBM_DepthType)

        # 获取SGBM参数
        min_disparity = self.ui.spinBox_SGBM_minDisparity.value()
        num_disparities = self.ui.spinBox_SGBM_numDisparities.value()
        block_size = self.ui.spinBox_SGBM_blockSize.value()
        p1 = self.ui.spinBox_SGBM_P1.value()
        p2 = self.ui.spinBox_SGBM_P2.value()
        disp12_max_diff = self.ui.spinBox_SGBM_disp12MaxDiff.value()
        uniqueness_ratio = self.ui.spinBox_SGBM_uniquenessRatio.value()
        speckle_window_size = self.ui.spinBox_SGBM_speckleWindowSize.value()
        speckle_range = self.ui.spinBox_SGBM_speckleRange.value()
        pre_filter_cap = self.ui.spinBox_SGBM_preFilterCap.value()
        mode_text = self.ui.comboBox_SGBM_mode.currentText()
        mode_value = self.sgbm_mode_map.get(mode_text, cv2.StereoSGBM_MODE_SGBM)

        print(f"SGBM参数: num_disparities={num_disparities}, block_size={block_size}")
        print(f"SGBM参数: P1={p1}, P2={p2}, mode={mode_text}")
        print(f"深度计算方式: {depth_type}")

        # 开始计时
        start_time = time.time()

        try:
            # 确保块大小是奇数
            if block_size % 2 == 0:
                block_size += 1
                self.ui.spinBox_SGBM_blockSize.setValue(block_size)

            # 创建SGBM对象
            stereo = cv2.StereoSGBM_create(
                minDisparity=min_disparity,
                numDisparities=num_disparities,
                blockSize=block_size,
                P1=p1,
                P2=p2,
                disp12MaxDiff=disp12_max_diff,
                uniquenessRatio=uniqueness_ratio,
                speckleWindowSize=speckle_window_size,
                speckleRange=speckle_range,
                preFilterCap=pre_filter_cap,
                mode=mode_value
            )

            # 计算视差
            print("正在计算视差...")
            disparity = stereo.compute(self.left_image, self.right_image)
            # 转换为浮点数
            disparity = disparity.astype(np.float32) / 16.0
            print("视差计算完成")

            # 计算深度图
            self.sgbm_depth = self.compute_depth_from_disparity(disparity, use_calib)

            self.sgbm_time = time.time() - start_time

            # 显示结果
            self.display_depth_result(self.sgbm_depth, self.ui.graphicsView_SGBM)
            self.ui.label_SGBM_Time.setText(f"计算时间: {self.sgbm_time:.3f}秒")
            self.ui.label_SGBM_DepthInfo.setText(f"深度类型: {depth_type} ({depth_unit})")

            # QMessageBox.information(self.ui, "成功",
            #                         f"SGBM深度图生成完成\n耗时: {self.sgbm_time:.3f}秒\n深度类型: {depth_type}")

        except Exception as e:
            error_msg = f"SGBM计算失败: {str(e)}"
            print(error_msg)
            QMessageBox.critical(self.ui, "错误", error_msg)

    def compare_algorithms(self):
        """对比两种算法"""
        print("开始对比BM和SGBM算法")

        if not self.check_images_loaded() or not self.validate_images_size():
            return

        # 获取深度计算方式
        use_calib, depth_type, depth_unit = self.get_depth_calculation_type(
            self.ui.comboBox_Compare_DepthType)

        # 获取对比参数
        num_disparities = self.ui.spinBox_Compare_numDisparities.value()
        block_size = self.ui.spinBox_Compare_blockSize.value()
        uniqueness_ratio = self.ui.spinBox_Compare_uniquenessRatio.value()
        speckle_window_size = self.ui.spinBox_Compare_speckleWindowSize.value()

        print(f"对比参数: num_disparities={num_disparities}, block_size={block_size}")
        print(f"深度计算方式: {depth_type}")

        # 设置BM参数并计算
        self.ui.spinBox_BM_numDisparities.setValue(num_disparities)
        self.ui.spinBox_BM_blockSize.setValue(block_size)
        self.ui.spinBox_BM_uniquenessRatio.setValue(uniqueness_ratio)
        self.ui.spinBox_BM_speckleWindowSize.setValue(speckle_window_size)
        self.ui.comboBox_BM_DepthType.setCurrentIndex(
            self.ui.comboBox_Compare_DepthType.currentIndex())

        self.generate_bm_depth()

        # 设置SGBM参数并计算
        self.ui.spinBox_SGBM_numDisparities.setValue(num_disparities)
        self.ui.spinBox_SGBM_blockSize.setValue(block_size)
        self.ui.spinBox_SGBM_uniquenessRatio.setValue(uniqueness_ratio)
        self.ui.spinBox_SGBM_speckleWindowSize.setValue(speckle_window_size)
        self.ui.comboBox_SGBM_DepthType.setCurrentIndex(
            self.ui.comboBox_Compare_DepthType.currentIndex())

        self.generate_sgbm_depth()

        # 显示对比结果
        if self.bm_depth is not None:
            self.display_depth_result(self.bm_depth, self.ui.graphicsView_Compare_BM)
            self.ui.label_Compare_BM_Time.setText(f"计算时间: {self.bm_time:.3f}秒")
            self.ui.label_Compare_BM_Info.setText(f"深度类型: {depth_type}")

        if self.sgbm_depth is not None:
            self.display_depth_result(self.sgbm_depth, self.ui.graphicsView_Compare_SGBM)
            self.ui.label_Compare_SGBM_Time.setText(f"计算时间: {self.sgbm_time:.3f}秒")
            self.ui.label_Compare_SGBM_Info.setText(f"深度类型: {depth_type}")

        # 显示统计信息
        if self.bm_depth is not None and self.sgbm_depth is not None:
            stats_text = self.generate_comparison_stats()
            self.ui.textEdit_Compare_Stats.setText(stats_text)

        # QMessageBox.information(self.ui, "成功", "对比完成")

    def generate_comparison_stats(self):
        """生成对比统计信息"""
        if self.bm_depth is None or self.sgbm_depth is None:
            return "无数据"

        stats = "=== 算法对比统计 ===\n\n"

        # 计算时间对比
        stats += f"计算时间:\n"
        stats += f"  BM: {self.bm_time:.3f}秒\n"
        stats += f"  SGBM: {self.sgbm_time:.3f}秒\n\n"

        # 深度类型
        index = self.ui.comboBox_Compare_DepthType.currentIndex()
        if index == 0:
            depth_type = "相对深度"
        elif index == 1:
            depth_type = "绝对深度（手动参数）"
        else:
            depth_type = "绝对深度（标定文件）"
        stats += f"深度类型: {depth_type}\n\n"

        # 深度统计
        bm_valid = self.bm_depth[self.bm_depth > 0]
        sgbm_valid = self.sgbm_depth[self.sgbm_depth > 0]

        if len(bm_valid) > 0 and len(sgbm_valid) > 0:
            stats += "深度统计:\n"
            stats += f"  BM均值: {np.mean(bm_valid):.2f}\n"
            stats += f"  SGBM均值: {np.mean(sgbm_valid):.2f}\n"
            stats += f"  BM标准差: {np.std(bm_valid):.2f}\n"
            stats += f"  SGBM标准差: {np.std(sgbm_valid):.2f}\n\n"

            # 有效像素比例
            bm_ratio = len(bm_valid) / self.bm_depth.size
            sgbm_ratio = len(sgbm_valid) / self.sgbm_depth.size
            stats += f"有效像素比例:\n"
            stats += f"  BM: {bm_ratio:.1%}\n"
            stats += f"  SGBM: {sgbm_ratio:.1%}\n"

        return stats

    def save_result(self, algorithm):
        """保存结果"""
        print(f"保存{algorithm.upper()}结果")

        if algorithm == 'bm' and self.bm_depth is not None:
            depth_map = self.bm_depth
            compute_time = self.bm_time
            depth_type_index = self.ui.comboBox_BM_DepthType.currentIndex()
        elif algorithm == 'sgbm' and self.sgbm_depth is not None:
            depth_map = self.sgbm_depth
            compute_time = self.sgbm_time
            depth_type_index = self.ui.comboBox_SGBM_DepthType.currentIndex()
        else:
            QMessageBox.warning(self.ui, "警告", f"请先生成{algorithm.upper()}深度图")
            return

        # 深度类型文本
        depth_types = ["相对深度（归一化）", "绝对深度（手动参数）", "绝对深度（标定文件）"]
        depth_type = depth_types[depth_type_index] if depth_type_index < len(depth_types) else "未知"

        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self.ui, f"保存{algorithm.upper()}深度图", "",
            "PNG图像 (*.png);;JPEG图像 (*.jpg);;所有文件 (*)")

        if file_path:
            # 确保文件有正确的扩展名
            if not file_path.endswith(('.png', '.jpg', '.jpeg')):
                file_path += '.png'

            # 保存深度图
            depth_display = depth_map.astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            cv2.imwrite(file_path, depth_colored)

            # 保存参数信息
            param_file = file_path.rsplit('.', 1)[0] + '_params.txt'
            with open(param_file, 'w') as f:
                f.write(f"算法: {algorithm.upper()}\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"深度计算方式: {depth_type}\n")
                if depth_type_index > 0:  # 绝对深度
                    f.write(f"焦距: {self.focal_length}\n")
                    f.write(f"基线: {self.baseline}\n")
                f.write(f"计算时间: {compute_time:.3f}秒\n")
                if algorithm == 'bm':
                    f.write(f"BM参数:\n")
                    f.write(f"  视差范围: {self.ui.spinBox_BM_numDisparities.value()}\n")
                    f.write(f"  块大小: {self.ui.spinBox_BM_blockSize.value()}\n")
                elif algorithm == 'sgbm':
                    f.write(f"SGBM参数:\n")
                    f.write(f"  视差范围: {self.ui.spinBox_SGBM_numDisparities.value()}\n")
                    f.write(f"  块大小: {self.ui.spinBox_SGBM_blockSize.value()}\n")
                    f.write(f"  P1: {self.ui.spinBox_SGBM_P1.value()}\n")
                    f.write(f"  P2: {self.ui.spinBox_SGBM_P2.value()}\n")

            QMessageBox.information(self.ui, "成功",
                                    f"{algorithm.upper()}深度图已保存\n{file_path}")

    def save_all_results(self):
        """保存所有结果"""
        print("保存所有结果")

        if self.bm_depth is None or self.sgbm_depth is None:
            QMessageBox.warning(self.ui, "警告", "请先生成对比结果")
            return

        # 保存BM结果
        self.save_result('bm')

        # 保存SGBM结果
        self.save_result('sgbm')

        QMessageBox.information(self.ui, "成功", "所有结果已保存")

    def display_depth_result(self, depth_map, graphics_view):
        """在QGraphicsView中显示深度图 - 紧凑版本"""
        if depth_map is None:
            graphics_view.setScene(None)
            return

        # 归一化深度图用于显示
        depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_display = depth_display.astype(np.uint8)

        # 计算缩放比例，使图像适合显示区域
        view_size = graphics_view.size()
        if view_size.width() > 0 and view_size.height() > 0:
            # 应用颜色映射
            depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

            # 获取视图尺寸，调整图像大小以适合
            target_width = max(100, view_size.width() - 20)  # 留出边距
            target_height = max(100, view_size.height() - 20)

            # 保持宽高比
            h, w = depth_colored.shape[:2]
            aspect_ratio = w / h

            if target_width / aspect_ratio <= target_height:
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                new_height = target_height
                new_width = int(target_height * aspect_ratio)

            # 调整图像大小
            if new_width > 0 and new_height > 0:
                depth_resized = cv2.resize(depth_colored, (new_width, new_height))

                # 转换为QImage
                h, w, ch = depth_resized.shape
                bytes_per_line = ch * w
                qimage = QImage(depth_resized.data, w, h, bytes_per_line, QImage.Format_RGB888)

                # 创建场景并显示
                scene = QGraphicsScene()
                pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(qimage))
                scene.addItem(pixmap_item)
                graphics_view.setScene(scene)
                graphics_view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
                graphics_view.setSceneRect(scene.sceneRect())

# 主程序
app = QApplication(sys.argv)

# 创建并显示主窗口
tool = StereoVisionTool()
if tool.ui:
    tool.ui.show()
else:
    QMessageBox.critical(None, "错误", "无法创建应用程序窗口")

app.exec()