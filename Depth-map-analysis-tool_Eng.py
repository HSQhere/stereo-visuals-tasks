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
        # Load UI from file
        ui_file_path = "Depth-map-analysis tool.ui"

        # Check if file exists
        if not os.path.exists(ui_file_path):
            QMessageBox.critical(None, "Error", f"Cannot find UI file: {ui_file_path}")
            return

        print(f"Loading UI file: {ui_file_path}")
        ui_file = QFile(ui_file_path)
        if not ui_file.open(QIODevice.ReadOnly):
            QMessageBox.critical(None, "Error", f"Cannot open UI file: {ui_file.errorString()}")
            return

        # Dynamically create window object from UI definition
        loader = QUiLoader()
        self.ui = loader.load(ui_file)
        ui_file.close()

        if not self.ui:
            QMessageBox.critical(None, "Error", "UI file loading failed")
            return

        # Set window title
        self.ui.setWindowTitle("Stereo Vision Depth Estimation Tool")

        # Apply compact style
        self.apply_qss_style()

        # Add tooltips for all parameters
        self.setup_tooltips()

        # Initialize variables
        self.left_image = None
        self.right_image = None
        self.calib_file = None
        self.calib_data = None

        # Default focal length and baseline
        self.focal_length = 1000.0
        self.baseline = 0.54

        # Result storage
        self.bm_depth = None
        self.bm_time = 0
        self.sgbm_depth = None
        self.sgbm_time = 0

        # SGBM mode mapping
        self.sgbm_mode_map = {
            "SGBM": cv2.StereoSGBM_MODE_SGBM,
            "SGBM 3WAY": cv2.StereoSGBM_MODE_SGBM_3WAY,
            "HH": cv2.StereoSGBM_MODE_HH
        }

        # Connect signals and slots
        self.connect_signals()
        print("UI loading completed")

    def apply_qss_style(self):
        """Apply QSS stylesheet to make boxes and buttons more visible"""
        qss = """
        /* Main window style */
        QMainWindow {
            background-color: #f5f5f5;
        }

        /* Group box style - more visible */
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

        /* Main operation buttons - more visible */
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

        /* Generate button special style */
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

        /* Save button special style */
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

        /* Load/Browse button */
        QPushButton[name*="Load"] {
            background-color: #9b59b6;
            border-color: #8e44ad;
        }

        QPushButton[name*="Load"]:hover {
            background-color: #8e44ad;
            border-color: #7d3c98;
        }

        /* Preview button */
        QPushButton[name*="Show"] {
            background-color: #f39c12;
            border-color: #e67e22;
        }

        QPushButton[name*="Show"]:hover {
            background-color: #e67e22;
            border-color: #d35400;
        }

        /* Tab widget style */
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

        /* Input box style */
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

        /* Label style */
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

        /* Graphics view style */
        QGraphicsView {
            border: 2px solid #7f8c8d;
            border-radius: 4px;
            background-color: #ecf0f1;
        }

        /* Text edit box style */
        QTextEdit {
            border: 1px solid #bdc3c7;
            border-radius: 3px;
            background-color: white;
            font-family: "Courier New", monospace;
            font-size: 10px;
        }

        /* Preview label - white background, dark text */
        QLabel#label_Preview {
            border: 2px solid #3498db;
            border-radius: 4px;
            background-color: white;
            color: #2c3e50;
            font-weight: bold;
            padding: 2px;
            font-size: 11px;
        }

        /* Sidebar group box special style */
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

        /* Comparison statistics group box special style */
        #groupBox_Compare_Stats {
            border: 2px solid #f39c12;
        }

        #groupBox_Compare_Stats::title {
            background-color: #f39c12;
        }

        /* BM related group box special style */
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

        /* SGBM related group box special style */
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

        /* Comparison parameters group box */
        #groupBox_Compare_Params {
            border: 2px solid #f39c12;
        }

        #groupBox_Compare_Params::title {
            background-color: #f39c12;
        }
        """

        # Apply QSS style
        self.ui.setStyleSheet(qss)

        # Set more special style for preview label - white background, blue border
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

    def setup_tooltips(self):
        """Set tooltips for all parameter controls"""
        # BM algorithm parameters
        self.ui.spinBox_BM_numDisparities.setToolTip(
            "Disparity range: must be an integer divisible by 16\n"
            "Defines the matching search range, larger values detect larger depth ranges but increase computation time\n"
            "Usually set to multiples of 16, such as 16, 32, 48, etc."
        )
        self.ui.spinBox_BM_blockSize.setToolTip(
            "Block size: window size for matching, must be odd\n"
            "Larger values provide better smoothing but lose more details\n"
            "Smaller values preserve details better but have more noise\n"
            "Recommended range: 5-21"
        )
        self.ui.spinBox_BM_textureThreshold.setToolTip(
            "Texture threshold: used to detect texture-deficient areas\n"
            "When texture in the window is insufficient, disparity is not calculated for that point\n"
            "Larger values make it easier to judge as texture-deficient areas\n"
            "Usually set to 5-20"
        )
        self.ui.spinBox_BM_uniquenessRatio.setToolTip(
            "Uniqueness ratio: measures the uniqueness of the best match\n"
            "Larger values impose stricter matching requirements, reducing false matches\n"
            "Usually set to 5-15"
        )
        self.ui.spinBox_BM_speckleWindowSize.setToolTip(
            "Speckle window: smoothing window size for post-processing\n"
            "Larger values provide stronger smoothing\n"
            "Set to 0 to disable speckle filter"
        )
        self.ui.spinBox_BM_speckleRange.setToolTip(
            "Speckle range: disparity change threshold for speckle filter\n"
            "Larger values allow greater disparity changes\n"
            "Usually set to 16 or 32"
        )
        self.ui.spinBox_BM_disp12MaxDiff.setToolTip(
            "Disparity difference: maximum allowed difference for left-right consistency check\n"
            "Used to detect false matches\n"
            "Larger values indicate higher tolerance\n"
            "-1 disables left-right consistency check"
        )
        self.ui.spinBox_BM_preFilterSize.setToolTip(
            "Pre-filter size: window size for pre-processing filter\n"
            "Must be odd, usually between 5-255\n"
            "Used for image pre-filtering"
        )
        self.ui.spinBox_BM_preFilterCap.setToolTip(
            "Pre-filter cap: truncation value for pre-processing filter\n"
            "Usually between 1-31\n"
            "Larger values provide stronger contrast enhancement"
        )
        self.ui.comboBox_BM_DepthType.setToolTip(
            "Depth calculation method:\n"
            "1. Relative depth (normalized) - only shows relative depth information, no actual physical units\n"
            "2. Absolute depth (manual parameters) - calculates actual depth using manually entered focal length and baseline\n"
            "3. Absolute depth (calibration file) - calculates actual depth using parameters from calibration file"
        )

        # SGBM algorithm parameters
        self.ui.spinBox_SGBM_minDisparity.setToolTip(
            "Minimum disparity: starting disparity value for search\n"
            "Can be negative, indicating negative disparity search\n"
            "Usually set to 0"
        )
        self.ui.spinBox_SGBM_numDisparities.setToolTip(
            "Disparity range: must be an integer divisible by 16\n"
            "Defines the matching search range\n"
            "Usually set to multiples of 16"
        )
        self.ui.spinBox_SGBM_blockSize.setToolTip(
            "Block size: window size for matching, must be odd\n"
            "SGBM recommends using smaller values, such as 3, 5, 7"
        )
        self.ui.spinBox_SGBM_P1.setToolTip(
            "P1 parameter: first penalty parameter controlling disparity smoothness\n"
            "P1 is the penalty when disparity change between adjacent pixels is 1\n"
            "Usually set to α * number of channels * block size²\n"
            "Usually α = 4, 8, 16"
        )
        self.ui.spinBox_SGBM_P2.setToolTip(
            "P2 parameter: second penalty parameter controlling disparity smoothness\n"
            "P2 is the penalty when disparity change between adjacent pixels is greater than 1\n"
            "Usually set to β * number of channels * block size²\n"
            "Usually β = 16, 32, 64, 96\n"
            "P2 should be greater than P1"
        )
        self.ui.spinBox_SGBM_disp12MaxDiff.setToolTip(
            "Disparity difference: maximum allowed difference for left-right consistency check\n"
            "Used to detect false matches\n"
            "-1 disables the check"
        )
        self.ui.spinBox_SGBM_uniquenessRatio.setToolTip(
            "Uniqueness ratio: measures the uniqueness of the best match\n"
            "Usually set to 5-15"
        )
        self.ui.spinBox_SGBM_speckleWindowSize.setToolTip(
            "Speckle window: smoothing window size for post-processing\n"
            "Set to 0 to disable speckle filter"
        )
        self.ui.spinBox_SGBM_speckleRange.setToolTip(
            "Speckle range: disparity change threshold for speckle filter\n"
            "Usually set to 16 or 32"
        )
        self.ui.spinBox_SGBM_preFilterCap.setToolTip(
            "Pre-filter cap: truncation value for pre-processing filter\n"
            "Usually between 1-63"
        )
        self.ui.comboBox_SGBM_mode.setToolTip(
            "Algorithm mode:\n"
            "SGBM - standard SGBM algorithm\n"
            "SGBM 3WAY - uses 3-way dynamic programming, faster\n"
            "HH - uses more precise but slower full dynamic programming"
        )
        self.ui.comboBox_SGBM_DepthType.setToolTip(
            "Depth calculation method:\n"
            "1. Relative depth (normalized) - only shows relative depth information\n"
            "2. Absolute depth (manual parameters) - calculates actual depth using manual parameters\n"
            "3. Absolute depth (calibration file) - calculates actual depth using calibration file parameters"
        )

        # Comparison parameters
        self.ui.spinBox_Compare_numDisparities.setToolTip(
            "Disparity range: uniformly applied to both BM and SGBM"
        )
        self.ui.spinBox_Compare_blockSize.setToolTip(
            "Block size: uniformly applied to both BM and SGBM"
        )
        self.ui.spinBox_Compare_uniquenessRatio.setToolTip(
            "Uniqueness ratio: uniformly applied to both BM and SGBM"
        )
        self.ui.spinBox_Compare_speckleWindowSize.setToolTip(
            "Speckle window: uniformly applied to both BM and SGBM"
        )
        self.ui.comboBox_Compare_DepthType.setToolTip(
            "Depth calculation method: uniformly applied to comparative analysis"
        )

        # Calibration parameters
        self.ui.doubleSpinBox_FocalLength.setToolTip(
            "Focal length: camera focal length in pixels\n"
            "Used to calculate absolute depth from disparity\n"
            "Typical values: 500-2000 pixels"
        )
        self.ui.doubleSpinBox_Baseline.setToolTip(
            "Baseline: distance between two cameras in meters\n"
            "Used to calculate absolute depth from disparity\n"
            "Typical values: 0.1-1.0 meters"
        )

        # Image input
        self.ui.lineEdit_LeftImage.setToolTip("Left camera image file path")
        self.ui.lineEdit_RightImage.setToolTip("Right camera image file path")
        self.ui.lineEdit_CalibFile.setToolTip(
            "Camera calibration file path (optional)\nSupports .txt, .yaml, .yml formats")

        # Buttons
        self.ui.pushButton_BM_Generate.setToolTip("Generate depth map using BM algorithm")
        self.ui.pushButton_SGBM_Generate.setToolTip("Generate depth map using SGBM algorithm")
        self.ui.pushButton_Compare_Generate.setToolTip("Compare two algorithms using same parameters")
        self.ui.pushButton_BM_Save.setToolTip("Save depth map generated by BM algorithm")
        self.ui.pushButton_SGBM_Save.setToolTip("Save depth map generated by SGBM algorithm")
        self.ui.pushButton_Compare_SaveAll.setToolTip("Save all comparison results")
        self.ui.pushButton_ShowLeft.setToolTip("Show left image in preview area")
        self.ui.pushButton_ShowRight.setToolTip("Show right image in preview area")

        # Depth map display areas
        self.ui.graphicsView_BM.setToolTip("Depth map display area generated by BM algorithm")
        self.ui.graphicsView_SGBM.setToolTip("Depth map display area generated by SGBM algorithm")
        self.ui.graphicsView_Compare_BM.setToolTip("BM depth map in comparative analysis")
        self.ui.graphicsView_Compare_SGBM.setToolTip("SGBM depth map in comparative analysis")

        print("Tooltip setup completed")

    def connect_signals(self):
        """Connect all signals and slots"""
        # Image loading buttons
        self.ui.pushButton_LoadLeft.clicked.connect(self.load_left_image)
        self.ui.pushButton_LoadRight.clicked.connect(self.load_right_image)
        self.ui.pushButton_LoadCalib.clicked.connect(self.load_calib_file)

        # Preview buttons
        self.ui.pushButton_ShowLeft.clicked.connect(lambda: self.show_image_preview('left'))
        self.ui.pushButton_ShowRight.clicked.connect(lambda: self.show_image_preview('right'))

        # Calibration parameter updates
        self.ui.doubleSpinBox_FocalLength.valueChanged.connect(self.update_focal_length)
        self.ui.doubleSpinBox_Baseline.valueChanged.connect(self.update_baseline)

        # BM tab
        self.ui.pushButton_BM_Generate.clicked.connect(self.generate_bm_depth)
        self.ui.pushButton_BM_Save.clicked.connect(lambda: self.save_result('bm'))

        # SGBM tab
        self.ui.pushButton_SGBM_Generate.clicked.connect(self.generate_sgbm_depth)
        self.ui.pushButton_SGBM_Save.clicked.connect(lambda: self.save_result('sgbm'))

        # Comparison tab
        self.ui.pushButton_Compare_Generate.clicked.connect(self.compare_algorithms)
        self.ui.pushButton_Compare_SaveAll.clicked.connect(self.save_all_results)

        print("All signal connections completed")

    def update_focal_length(self, value):
        """Update focal length"""
        self.focal_length = value
        print(f"Focal length updated to: {value}")

    def update_baseline(self, value):
        """Update baseline"""
        self.baseline = value
        print(f"Baseline updated to: {value}")

    def parse_calib_file(self, file_path):
        """Parse calibration file - compatible with multiple formats"""
        import numpy as np

        focal_from_file = 0.0  # Use temporary variables
        baseline_from_file = 0.0

        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Parse file content
        for line in lines:
            line = line.strip()

            # Format 1: KITTI format
            if line.startswith('P2:'):
                matrix_str = line[3:].strip()
                matrix = np.array(matrix_str.split()).astype('float32').reshape(3, 4)
                focal_from_file = matrix[0, 0]

            elif line.startswith('P3:'):
                matrix_str = line[3:].strip()
                matrix = np.array(matrix_str.split()).astype('float32').reshape(3, 4)
                baseline_from_file = -matrix[0, 3] / matrix[0, 0]

            # Format 2: Camera matrix format
            elif line.startswith('cam0=') or line.startswith('cam1='):
                matrix_part = line.split('=', 1)[1]
                matrix_str = matrix_part.replace('[', '').replace(']', '')
                values = matrix_str.replace(';', ' ').split()
                if values:
                    focal_from_file = float(values[0])

            # Baseline information
            elif line.startswith('baseline='):
                # Extract value after equals sign and convert to meters
                value_str = line.split('=', 1)[1].strip()
                baseline_mm = float(value_str)  # Read as millimeters
                baseline_from_file = baseline_mm / 1000.0  # Convert to meters

            # Format 3: Middlebury format - look for specific patterns
            elif 'fx' in line.lower():
                # Handle focal length in Middlebury format
                parts = line.split('=')
                if len(parts) == 2:
                    try:
                        focal_from_file = float(parts[1].strip())
                    except ValueError:
                        pass

            elif 'baseline' in line.lower() and '=' in line:
                # Handle baseline in various formats
                parts = line.split('=')
                if len(parts) == 2:
                    try:
                        baseline_val = float(parts[1].strip())
                        # Check if baseline is in mm and convert to meters
                        if baseline_val < 10:  # Likely in meters
                            baseline_from_file = baseline_val
                        else:  # Likely in mm
                            baseline_from_file = baseline_val / 1000.0
                    except ValueError:
                        pass

        # Update UI and class variables
        if focal_from_file > 0:
            self.ui.doubleSpinBox_FocalLength.setValue(focal_from_file)
            self.focal_length = focal_from_file  # Update class variable directly

        if baseline_from_file > 0:
            self.ui.doubleSpinBox_Baseline.setValue(baseline_from_file)
            self.baseline = baseline_from_file  # Update class variable directly

        # Display results
        if focal_from_file > 0 or baseline_from_file > 0:
            QMessageBox.information(
                self.ui,
                "Calibration Result",
                f"Parameters loaded from calibration file:\n\n"
                f"Focal length: {self.focal_length:.2f} pixels\n"
                f"Baseline: {self.baseline:.4f} meters"
            )
        else:
            QMessageBox.warning(
                self.ui,
                "Calibration Warning",
                "No valid calibration parameters found in the file.\n"
                "Using default or manual parameters instead."
            )

        print(f"Calibration loaded: Focal={self.focal_length}, Baseline={self.baseline}")

    def load_left_image(self):
        """Load left image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.ui, "Select Left Image", "",
            "Image files (*.png *.jpg *.jpeg *.bmp *.tiff)")

        if file_path:
            self.ui.lineEdit_LeftImage.setText(file_path)
            self.left_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if self.left_image is not None:
                print(f"Left image loaded successfully: {Path(file_path).name}, size: {self.left_image.shape}")
                self.show_image_preview('left')
                self.update_image_info()
            else:
                QMessageBox.warning(self.ui, "Error", "Cannot load left image")

    def load_right_image(self):
        """Load right image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.ui, "Select Right Image", "",
            "Image files (*.png *.jpg *.jpeg *.bmp *.tiff)")

        if file_path:
            self.ui.lineEdit_RightImage.setText(file_path)
            self.right_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if self.right_image is not None:
                print(f"Right image loaded successfully: {Path(file_path).name}, size: {self.right_image.shape}")
                self.show_image_preview('right')
                self.update_image_info()
            else:
                QMessageBox.warning(self.ui, "Error", "Cannot load right image")

    def load_calib_file(self):
        """Load calibration file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.ui, "Select Calibration File", "",
            "Calibration files (*.txt *.yaml *.yml);;All files (*)")

        if file_path:
            self.ui.lineEdit_CalibFile.setText(file_path)
            self.calib_file = file_path
            self.parse_calib_file(file_path)

    def show_image_preview(self, image_type):
        """Show image preview"""
        if image_type == 'left' and self.left_image is not None:
            self.display_image(self.left_image, self.ui.label_Preview)
        elif image_type == 'right' and self.right_image is not None:
            self.display_image(self.right_image, self.ui.label_Preview)
        else:
            self.ui.label_Preview.setText("No image")
            self.ui.label_Preview.setPixmap(QPixmap())

    def update_image_info(self):
        """Update image information display"""
        if self.left_image is not None:
            h, w = self.left_image.shape
            image_info = f"Image size: {w}×{h}"

            # Update image information in all tabs
            self.ui.label_BM_ImageInfo.setText(image_info)
            self.ui.label_SGBM_ImageInfo.setText(image_info)

    def display_image(self, image, label_widget):
        """Display image in QLabel - compact version"""
        if image is None:
            return

        # Convert OpenCV image to QImage
        if len(image.shape) == 2:  # Grayscale image
            h, w = image.shape
            bytes_per_line = w
            qimage = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:  # Color image
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qimage = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Get label size, adjust image size
        label_size = label_widget.size()
        if label_size.width() > 0 and label_size.height() > 0:
            # Calculate size suitable for label
            target_width = label_size.width() - 4  # Leave border
            target_height = label_size.height() - 4

            # Maintain aspect ratio
            aspect_ratio = w / h
            if target_width / aspect_ratio <= target_height:
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                new_height = target_height
                new_width = int(target_height * aspect_ratio)

            # Create thumbnail
            qimage_scaled = qimage.scaled(
                new_width, new_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            label_widget.setPixmap(QPixmap.fromImage(qimage_scaled))
            label_widget.setScaledContents(False)  # Do not stretch image

    def check_images_loaded(self):
        """Check if images are loaded"""
        if self.left_image is None:
            QMessageBox.warning(self.ui, "Error", "Please load left image first")
            return False
        if self.right_image is None:
            QMessageBox.warning(self.ui, "Error", "Please load right image first")
            return False
        return True

    def validate_images_size(self):
        """Verify image sizes are consistent"""
        if self.left_image.shape != self.right_image.shape:
            error_msg = f"Image sizes inconsistent: left{self.left_image.shape}, right{self.right_image.shape}"
            QMessageBox.warning(self.ui, "Error", error_msg)
            return False
        return True

    def get_depth_calculation_type(self, combo_box):
        """Get depth calculation type"""
        index = combo_box.currentIndex()
        if index == 0:
            return False, "Relative depth (normalized)", "normalized value"
        elif index == 1:
            return True, "Absolute depth (manual parameters)", "meters"
        elif index == 2:
            return True, "Absolute depth (calibration file)", "meters"
        else:
            return False, "Relative depth (normalized)", "normalized value"

    def compute_depth_from_disparity(self, disparity, use_calib, normalization_method='percentile'):
        """Calculate depth map from disparity map"""
        if disparity is None:
            return None

        # Avoid division by zero
        disparity_filtered = disparity.copy().astype(np.float32)
        disparity_filtered[disparity_filtered <= 0] = 0.1

        if use_calib:
            # Calculate absolute depth using calibration parameters
            depth_map = (self.focal_length * self.baseline) / disparity_filtered

            # Limit maximum depth
            max_depth = 100.0  # meters
            depth_map[depth_map > max_depth] = max_depth
            depth_map[depth_map < 0] = 0

            print(f"Calculating absolute depth, focal length={self.focal_length}, baseline={self.baseline}")

            return depth_map
        else:
            # Relative depth calculation
            # Step 1: Analyze disparity distribution
            print("=== Disparity Analysis ===")
            print(f"Disparity shape: {disparity_filtered.shape}")
            print(f"Disparity range: [{disparity_filtered.min():.2f}, {disparity_filtered.max():.2f}]")

            # Step 2: Handle zero and near-zero values
            min_valid_disparity = 0.5  # Minimum valid disparity threshold
            mask_valid = disparity_filtered > min_valid_disparity

            if not np.any(mask_valid):
                print(f"Warning: No disparity values > {min_valid_disparity}")
                return np.zeros_like(disparity_filtered)

            # Statistics before filtering
            total_pixels = disparity_filtered.size
            valid_pixels = np.sum(mask_valid)
            zero_pixels = np.sum(disparity_filtered <= min_valid_disparity)

            print(f"Total pixels: {total_pixels}")
            print(f"Valid pixels (> {min_valid_disparity}): {valid_pixels} ({valid_pixels / total_pixels * 100:.1f}%)")
            print(f"Invalid pixels (<= {min_valid_disparity}): {zero_pixels} ({zero_pixels / total_pixels * 100:.1f}%)")

            # Step 3: Calculate relative depth
            relative_depth = np.zeros_like(disparity_filtered)
            relative_depth[mask_valid] = 1.0 / disparity_filtered[mask_valid]
            valid_values = relative_depth[mask_valid]

            print(f"Relative depth range (valid): [{valid_values.min():.4f}, {valid_values.max():.4f}]")
            print(f"Relative depth mean: {valid_values.mean():.4f}, std: {valid_values.std():.4f}")

            # Step 4: Apply normalization method
            if normalization_method == 'percentile':
                # Method 1: Percentile clipping (most robust)
                p1, p99 = np.percentile(valid_values, [1, 99])
                print(f"Percentile 1%: {p1:.4f}, 99%: {p99:.4f}")
                depth_clipped = np.clip(relative_depth, p1, p99)
                depth_normalized = (depth_clipped - p1) / (p99 - p1 + 1e-10)

            elif normalization_method == 'minmax':
                # Method 2: Min-max normalization
                min_val, max_val = np.min(valid_values), np.max(valid_values)
                print(f"Min: {min_val:.4f}, Max: {max_val:.4f}")
                depth_normalized = (relative_depth - min_val) / (max_val - min_val + 1e-10)

            elif normalization_method == 'log':
                # Method 3: Log normalization (compress high dynamic range)
                epsilon = 1e-6
                log_depth = np.log(relative_depth + epsilon)
                log_valid = log_depth[mask_valid]
                p1, p99 = np.percentile(log_valid, [1, 99])
                print(f"Log percentile 1%: {p1:.4f}, 99%: {p99:.4f}")
                log_clipped = np.clip(log_depth, p1, p99)
                depth_normalized = (log_clipped - p1) / (p99 - p1 + 1e-10)

            elif normalization_method == 'adaptive':
                # Method 4: Adaptive histogram equalization
                p5, p95 = np.percentile(valid_values, [5, 95])
                depth_clipped = np.clip(relative_depth, p5, p95)
                depth_temp = ((depth_clipped - p5) / (p95 - p5 + 1e-10) * 255).astype(np.uint8)

                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                depth_enhanced = clahe.apply(depth_temp)
                depth_normalized = depth_enhanced.astype(np.float32) / 255.0

            # Step 5: Convert to 0-255 range
            depth_map = depth_normalized * 255
            depth_map = np.clip(depth_map, 0, 255).astype(np.uint8)

            # Apply median filter to reduce salt-and-pepper noise
            if valid_pixels > 100:
                depth_map = cv2.medianBlur(depth_map, 3)

            print("=== Depth Generation Complete ===")
            print(f"Depth map range: [{depth_map.min()}, {depth_map.max()}]")
            print(f"Depth map dtype: {depth_map.dtype}")

            return depth_map.astype(np.float32)

    def generate_bm_depth(self):
        """Generate BM depth map"""
        print("Starting BM depth map generation")
        if not self.check_images_loaded() or not self.validate_images_size():
            return

        # Get depth calculation method
        use_calib, depth_type, depth_unit = self.get_depth_calculation_type(
            self.ui.comboBox_BM_DepthType)

        # Get BM parameters
        num_disparities = self.ui.spinBox_BM_numDisparities.value()
        block_size = self.ui.spinBox_BM_blockSize.value()
        texture_threshold = self.ui.spinBox_BM_textureThreshold.value()
        uniqueness_ratio = self.ui.spinBox_BM_uniquenessRatio.value()
        speckle_window_size = self.ui.spinBox_BM_speckleWindowSize.value()
        speckle_range = self.ui.spinBox_BM_speckleRange.value()
        disp12_max_diff = self.ui.spinBox_BM_disp12MaxDiff.value()
        pre_filter_size = self.ui.spinBox_BM_preFilterSize.value()
        pre_filter_cap = self.ui.spinBox_BM_preFilterCap.value()

        print(f"BM parameters: num_disparities={num_disparities}, block_size={block_size}")
        print(f"BM parameters: texture_threshold={texture_threshold}, uniqueness_ratio={uniqueness_ratio}")
        print(f"Depth calculation method: {depth_type}")

        # Start timing
        start_time = time.time()

        try:
            # Ensure block size is odd
            if block_size % 2 == 0:
                block_size += 1
                self.ui.spinBox_BM_blockSize.setValue(block_size)

            # Create BM object
            stereo = cv2.StereoBM_create(
                numDisparities=num_disparities,
                blockSize=block_size
            )

            # Set other parameters
            stereo.setTextureThreshold(texture_threshold)
            stereo.setUniquenessRatio(uniqueness_ratio)
            stereo.setSpeckleWindowSize(speckle_window_size)
            stereo.setSpeckleRange(speckle_range)
            stereo.setDisp12MaxDiff(disp12_max_diff)
            stereo.setPreFilterSize(pre_filter_size)
            stereo.setPreFilterCap(pre_filter_cap)

            # Calculate disparity
            print("Calculating disparity...")
            disparity = stereo.compute(self.left_image, self.right_image)
            print("Disparity calculation completed")

            # Calculate depth map
            self.bm_depth = self.compute_depth_from_disparity(disparity, use_calib)
            self.bm_time = time.time() - start_time

            # Display results
            self.display_depth_result(self.bm_depth, self.ui.graphicsView_BM)
            self.ui.label_BM_Time.setText(f"Calculation time: {self.bm_time:.3f} seconds")
            self.ui.label_BM_DepthInfo.setText(f"Depth type: {depth_type} ({depth_unit})")

        except Exception as e:
            error_msg = f"BM calculation failed: {str(e)}"
            print(error_msg)
            QMessageBox.critical(self.ui, "Error", error_msg)

    def generate_sgbm_depth(self):
        """Generate SGBM depth map"""
        print("Starting SGBM depth map generation")
        if not self.check_images_loaded() or not self.validate_images_size():
            return

        # Get depth calculation method
        use_calib, depth_type, depth_unit = self.get_depth_calculation_type(
            self.ui.comboBox_SGBM_DepthType)

        # Get SGBM parameters
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

        print(f"SGBM parameters: num_disparities={num_disparities}, block_size={block_size}")
        print(f"SGBM parameters: P1={p1}, P2={p2}, mode={mode_text}")
        print(f"Depth calculation method: {depth_type}")

        # Start timing
        start_time = time.time()

        try:
            # Ensure block size is odd
            if block_size % 2 == 0:
                block_size += 1
                self.ui.spinBox_SGBM_blockSize.setValue(block_size)

            # Create SGBM object
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

            # Calculate disparity
            print("Calculating disparity...")
            disparity = stereo.compute(self.left_image, self.right_image)
            # Convert to float
            disparity = disparity.astype(np.float32) / 16.0
            print("Disparity calculation completed")

            # Calculate depth map
            self.sgbm_depth = self.compute_depth_from_disparity(disparity, use_calib)
            self.sgbm_time = time.time() - start_time

            # Display results
            self.display_depth_result(self.sgbm_depth, self.ui.graphicsView_SGBM)
            self.ui.label_SGBM_Time.setText(f"Calculation time: {self.sgbm_time:.3f} seconds")
            self.ui.label_SGBM_DepthInfo.setText(f"Depth type: {depth_type} ({depth_unit})")

        except Exception as e:
            error_msg = f"SGBM calculation failed: {str(e)}"
            print(error_msg)
            QMessageBox.critical(self.ui, "Error", error_msg)

    def compare_algorithms(self):
        """Compare two algorithms"""
        print("Starting BM and SGBM algorithm comparison")
        if not self.check_images_loaded() or not self.validate_images_size():
            return

        # Get depth calculation method
        use_calib, depth_type, depth_unit = self.get_depth_calculation_type(
            self.ui.comboBox_Compare_DepthType)

        # Get comparison parameters
        num_disparities = self.ui.spinBox_Compare_numDisparities.value()
        block_size = self.ui.spinBox_Compare_blockSize.value()
        uniqueness_ratio = self.ui.spinBox_Compare_uniquenessRatio.value()
        speckle_window_size = self.ui.spinBox_Compare_speckleWindowSize.value()

        print(f"Comparison parameters: num_disparities={num_disparities}, block_size={block_size}")
        print(f"Depth calculation method: {depth_type}")

        # Set BM parameters and calculate
        self.ui.spinBox_BM_numDisparities.setValue(num_disparities)
        self.ui.spinBox_BM_blockSize.setValue(block_size)
        self.ui.spinBox_BM_uniquenessRatio.setValue(uniqueness_ratio)
        self.ui.spinBox_BM_speckleWindowSize.setValue(speckle_window_size)
        self.ui.comboBox_BM_DepthType.setCurrentIndex(
            self.ui.comboBox_Compare_DepthType.currentIndex())
        self.generate_bm_depth()

        # Set SGBM parameters and calculate
        self.ui.spinBox_SGBM_numDisparities.setValue(num_disparities)
        self.ui.spinBox_SGBM_blockSize.setValue(block_size)
        self.ui.spinBox_SGBM_uniquenessRatio.setValue(uniqueness_ratio)
        self.ui.spinBox_SGBM_speckleWindowSize.setValue(speckle_window_size)
        self.ui.comboBox_SGBM_DepthType.setCurrentIndex(
            self.ui.comboBox_Compare_DepthType.currentIndex())
        self.generate_sgbm_depth()

        # Display comparison results
        if self.bm_depth is not None:
            self.display_depth_result(self.bm_depth, self.ui.graphicsView_Compare_BM)
            self.ui.label_Compare_BM_Time.setText(f"Calculation time: {self.bm_time:.3f} seconds")
            self.ui.label_Compare_BM_Info.setText(f"Depth type: {depth_type}")

        if self.sgbm_depth is not None:
            self.display_depth_result(self.sgbm_depth, self.ui.graphicsView_Compare_SGBM)
            self.ui.label_Compare_SGBM_Time.setText(f"Calculation time: {self.sgbm_time:.3f} seconds")
            self.ui.label_Compare_SGBM_Info.setText(f"Depth type: {depth_type}")

        # Display statistical information
        if self.bm_depth is not None and self.sgbm_depth is not None:
            stats_text = self.generate_comparison_stats()
            self.ui.textEdit_Compare_Stats.setText(stats_text)

    def generate_comparison_stats(self):
        """Generate comparison statistical information"""
        if self.bm_depth is None or self.sgbm_depth is None:
            return "No data"

        stats = "=== Algorithm Comparison Statistics ===\n\n"

        # Time comparison
        stats += f"Calculation time:\n"
        stats += f" BM: {self.bm_time:.3f} seconds\n"
        stats += f" SGBM: {self.sgbm_time:.3f} seconds\n\n"

        # Depth type
        index = self.ui.comboBox_Compare_DepthType.currentIndex()
        if index == 0:
            depth_type = "Relative depth"
        elif index == 1:
            depth_type = "Absolute depth (manual parameters)"
        else:
            depth_type = "Absolute depth (calibration file)"

        stats += f"Depth type: {depth_type}\n\n"

        # Depth statistics
        bm_valid = self.bm_depth[self.bm_depth > 0]
        sgbm_valid = self.sgbm_depth[self.sgbm_depth > 0]

        if len(bm_valid) > 0 and len(sgbm_valid) > 0:
            stats += "Depth statistics:\n"
            stats += f" BM mean: {np.mean(bm_valid):.2f}\n"
            stats += f" SGBM mean: {np.mean(sgbm_valid):.2f}\n"
            stats += f" BM standard deviation: {np.std(bm_valid):.2f}\n"
            stats += f" SGBM standard deviation: {np.std(sgbm_valid):.2f}\n\n"

        # Valid pixel ratio
        bm_ratio = len(bm_valid) / self.bm_depth.size
        sgbm_ratio = len(sgbm_valid) / self.sgbm_depth.size

        stats += f"Valid pixel ratio:\n"
        stats += f" BM: {bm_ratio:.1%}\n"
        stats += f" SGBM: {sgbm_ratio:.1%}\n"

        return stats

    def save_result(self, algorithm):
        """Save result"""
        print(f"Saving {algorithm.upper()} result")

        if algorithm == 'bm' and self.bm_depth is not None:
            depth_map = self.bm_depth
            compute_time = self.bm_time
            depth_type_index = self.ui.comboBox_BM_DepthType.currentIndex()
        elif algorithm == 'sgbm' and self.sgbm_depth is not None:
            depth_map = self.sgbm_depth
            compute_time = self.sgbm_time
            depth_type_index = self.ui.comboBox_SGBM_DepthType.currentIndex()
        else:
            QMessageBox.warning(self.ui, "Warning",
                                f"Please generate {algorithm.upper()} depth map first")
            return

        # Depth type text
        depth_types = ["Relative depth (normalized)", "Absolute depth (manual parameters)",
                       "Absolute depth (calibration file)"]
        depth_type = depth_types[depth_type_index] if depth_type_index < len(depth_types) else "Unknown"

        # Select save path
        file_path, _ = QFileDialog.getSaveFileName(
            self.ui, f"Save {algorithm.upper()} Depth Map", "",
            "PNG image (*.png);;JPEG image (*.jpg);;All files (*)")

        if file_path:
            # Ensure file has correct extension
            if not file_path.endswith(('.png', '.jpg', '.jpeg')):
                file_path += '.png'

            # Save depth map
            depth_display = depth_map.astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            cv2.imwrite(file_path, depth_colored)

            # Save parameter information
            param_file = file_path.rsplit('.', 1)[0] + '_params.txt'
            with open(param_file, 'w') as f:
                f.write(f"Algorithm: {algorithm.upper()}\n")
                f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Depth calculation method: {depth_type}\n")
                if depth_type_index > 0:  # Absolute depth
                    f.write(f"Focal length: {self.focal_length}\n")
                    f.write(f"Baseline: {self.baseline}\n")
                f.write(f"Calculation time: {compute_time:.3f} seconds\n")
                if algorithm == 'bm':
                    f.write(f"BM parameters:\n")
                    f.write(f"  Disparity range: {self.ui.spinBox_BM_numDisparities.value()}\n")
                    f.write(f"  Block size: {self.ui.spinBox_BM_blockSize.value()}\n")
                elif algorithm == 'sgbm':
                    f.write(f"SGBM parameters:\n")
                    f.write(f"  Disparity range: {self.ui.spinBox_SGBM_numDisparities.value()}\n")
                    f.write(f"  Block size: {self.ui.spinBox_SGBM_blockSize.value()}\n")
                    f.write(f"  P1: {self.ui.spinBox_SGBM_P1.value()}\n")
                    f.write(f"  P2: {self.ui.spinBox_SGBM_P2.value()}\n")

            QMessageBox.information(self.ui, "Success",
                                    f"{algorithm.upper()} depth map saved\n{file_path}")

    def save_all_results(self):
        """Save all results"""
        print("Saving all results")
        if self.bm_depth is None or self.sgbm_depth is None:
            QMessageBox.warning(self.ui, "Warning", "Please generate comparison results first")
            return

        # Save BM result
        self.save_result('bm')

        # Save SGBM result
        self.save_result('sgbm')

        QMessageBox.information(self.ui, "Success", "All results saved")

    def display_depth_result(self, depth_map, graphics_view):
        """Display depth map in QGraphicsView """
        if depth_map is None:
            graphics_view.setScene(None)
            return

        # Normalize depth map for display
        depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_display = depth_display.astype(np.uint8)

        # Calculate scaling ratio to make image fit display area
        view_size = graphics_view.size()
        if view_size.width() > 0 and view_size.height() > 0:
            # Apply color map
            depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

            # Get view dimensions, adjust image size to fit
            target_width = max(100, view_size.width() - 20)  # Leave margins
            target_height = max(100, view_size.height() - 20)

            # Maintain aspect ratio
            h, w = depth_colored.shape[:2]
            aspect_ratio = w / h

            if target_width / aspect_ratio <= target_height:
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                new_height = target_height
                new_width = int(target_height * aspect_ratio)

            # Resize image
            if new_width > 0 and new_height > 0:
                depth_resized = cv2.resize(depth_colored, (new_width, new_height))

                # Convert to QImage
                h, w, ch = depth_resized.shape
                bytes_per_line = ch * w
                qimage = QImage(depth_resized.data, w, h, bytes_per_line, QImage.Format_RGB888)

                # Create scene and display
                scene = QGraphicsScene()
                pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(qimage))
                scene.addItem(pixmap_item)
                graphics_view.setScene(scene)
                graphics_view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
                graphics_view.setSceneRect(scene.sceneRect())


# Main program
app = QApplication(sys.argv)

# Create and show main window
tool = StereoVisionTool()
if tool.ui:
    tool.ui.show()
else:
    QMessageBox.critical(None, "Error", "Cannot create application window")

app.exec()
