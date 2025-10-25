import sys
import os
import cv2
import json
import yaml
import re
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QFileDialog, QMessageBox, QComboBox, QTextEdit, QProgressBar,
    QGroupBox, QGridLayout, QListWidget, QListWidgetItem, QSpinBox, QCheckBox, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

# ----------------- Logging & folders -----------------
os.makedirs('captures', exist_ok=True)
os.makedirs('predictions', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('config', exist_ok=True)
os.makedirs('assets', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FrozenStrawApp")

# ----------------- Default Config -----------------
DEFAULT_CONFIG = {
    'model_path': 'yolo11n.pt',
    'confidence_threshold': 0.25,
    'camera_check_indices': 10,
    'ai_stations': ['Station-1', 'Station-2', 'Station-3', 'Station-4'],
    'straw_sizes': ['0.25ml', '0.5ml'],
    'output_dirs': {
        'captures': 'captures',
        'predictions': 'predictions',
        'logs': 'logs'
    }
}

CONFIG_PATH = 'config/config.yaml'

def load_config() -> Dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                cfg = yaml.safe_load(f) or {}
                merged = DEFAULT_CONFIG.copy()
                merged.update(cfg)
                return merged
        except Exception as e:
            logger.error("Failed to load config.yaml: %s", e)
            return DEFAULT_CONFIG.copy()
    else:
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

def save_config(cfg: Dict):
    try:
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(cfg, f)
    except Exception as e:
        logger.error("Failed to save config.yaml: %s", e)

app_config = load_config()

# ----------------- Utilities -----------------
def sanitize_filename(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', str(text or ''))

def generate_filename(metadata: Dict) -> str:
    bull_name = sanitize_filename(metadata.get('bull_name', 'Unknown'))
    reg_no = sanitize_filename(metadata.get('reg_no', 'NoReg'))
    batch_id = sanitize_filename(metadata.get('batch_id', 'NoBatch'))
    order_id = sanitize_filename(metadata.get('order_id', 'NoOrder'))
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    return f"{bull_name}_{reg_no}_{batch_id}_{order_id}_{timestamp}"

def save_image_and_metadata(image: np.ndarray, metadata: Dict, camera_info: Dict, filename: str) -> str:
    img_dir = app_config['output_dirs']['captures']
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, f"{filename}.jpg")
    cv2.imwrite(img_path, image)
    metadata_copy = metadata.copy()
    metadata_copy['camera_used'] = camera_info
    metadata_copy['capture_timestamp'] = datetime.now().isoformat()
    metadata_copy['image_shape'] = image.shape
    json_path = os.path.join(img_dir, f"{filename}.json")
    with open(json_path, 'w') as f:
        json.dump(metadata_copy, f, indent=2)
    logger.info("Saved captured image and metadata: %s", filename)
    return img_path

def save_prediction_results(annotated_img: np.ndarray, detections: List[Dict], metadata: Dict, filename: str, inference_time: float) -> Tuple[str, str]:
    pred_dir = app_config['output_dirs']['predictions']
    os.makedirs(pred_dir, exist_ok=True)
    pred_img_path = os.path.join(pred_dir, f"{filename}_annotated.jpg")
    cv2.imwrite(pred_img_path, annotated_img)
    pred_json = {
        'metadata': metadata,
        'inference_time': inference_time,
        'model_used': app_config['model_path'],
        'confidence_threshold': app_config['confidence_threshold'],
        'detections': detections,
        'num_detections': len(detections),
        'prediction_timestamp': datetime.now().isoformat()
    }
    pred_json_path = os.path.join(pred_dir, f"{filename}.json")
    with open(pred_json_path, 'w') as f:
        json.dump(pred_json, f, indent=2)
    logger.info("Saved prediction results: %s", filename)
    return pred_img_path, pred_json_path

# ----------------- Camera Manager -----------------
class CameraManager:
    def __init__(self, max_indices: int = 10):
        self.cap = None
        self.camera_index = None
        self.max_indices = max_indices

    def discover_usb_cameras(self) -> List[Dict]:
        """Discover USB cameras compatible with DirectShow (Windows)"""
        cameras = []
        for i in range(self.max_indices):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        cameras.append({'index': i, 'name': f'USB Camera {i}', 'type': 'USB'})
                    cap.release()
            except Exception as e:
                logger.debug(f"Camera {i} check error: {e}")
        return cameras

    def open_camera(self, camera_info: Dict) -> bool:
        try:
            if camera_info['type'] == 'USB':
                idx = int(camera_info['index'])
                self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                
                # Set high resolution for capture
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5488)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3672)
                
                opened = self.cap.isOpened()
                if opened:
                    self.camera_index = idx
                    logger.info(f"Camera opened: index {idx}")
                return opened
            elif camera_info['type'] == 'RTSP':
                url = camera_info.get('url')
                self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                opened = self.cap.isOpened()
                return opened
            else:
                return False
        except Exception as e:
            logger.error("open_camera error: %s", e)
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        try:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    return frame
            return None
        except Exception as e:
            logger.error("read_frame error: %s", e)
            return None

    def release(self):
        try:
            if self.cap:
                self.cap.release()
            self.cap = None
            self.camera_index = None
        except Exception as e:
            logger.error("release error: %s", e)

# ----------------- YOLO Predictor -----------------
class YOLOPredictor:
    def __init__(self, model_path: str, confidence: float = 0.25):
        self.model = None
        self.model_path = model_path
        self.confidence = confidence
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info("Loaded YOLO model from %s", self.model_path)
        except Exception as e:
            logger.error("YOLO model load failed: %s", e)
            self.model = None

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict], float]:
        """Run inference synchronously and return annotated image, detections, inference_time"""
        if self.model is None:
            logger.error("Model not loaded")
            return image, [], 0.0
        try:
            start = time.time()
            results = self.model(image, conf=self.confidence)
            inference_time = time.time() - start

            detections = []
            annotated_img = image.copy()

            result = results[0]
            if hasattr(result, 'boxes'):
                boxes = getattr(result, 'boxes')
                names = result.names if hasattr(result, 'names') else {}
                for box in boxes:
                    try:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                    except Exception:
                        arr = box.xyxy
                        if isinstance(arr, (list, tuple, np.ndarray)):
                            x1, y1, x2, y2 = map(int, arr[:4])
                        else:
                            continue
                    try:
                        conf = float(box.conf[0].cpu().numpy())
                    except Exception:
                        conf = float(getattr(box, 'conf', 0.0))
                    try:
                        cls_idx = int(box.cls[0].cpu().numpy())
                    except Exception:
                        cls_idx = int(getattr(box, 'cls', 0))
                    class_name = names.get(cls_idx, str(cls_idx))
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated_img, label, (x1, max(y1 - 6, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                try:
                    annotated_img = result.plot()
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for b in result.boxes:
                            detections.append({
                                'class': str(getattr(b, 'cls', '')),
                                'confidence': float(getattr(b, 'conf', 0.0))
                            })
                except Exception as e:
                    logger.warning("Could not annotate using result.plot(): %s", e)

            logger.info("Inference done: %d detections in %.2fs", len(detections), inference_time)
            return annotated_img, detections, inference_time

        except Exception as e:
            logger.error("Predict error: %s\n%s", e, traceback.format_exc())
            return image, [], 0.0

# Worker thread for running predictions
class PredictionWorker(QThread):
    finished = pyqtSignal(np.ndarray, list, float, str)
    error = pyqtSignal(str)

    def __init__(self, predictor: YOLOPredictor, image: np.ndarray, metadata: Dict, camera_info: Dict):
        super().__init__()
        self.predictor = predictor
        self.image = image
        self.metadata = metadata
        self.camera_info = camera_info

    def run(self):
        try:
            filename = generate_filename(self.metadata)
            save_image_and_metadata(self.image, self.metadata, self.camera_info, filename)
            annotated_img, detections, inference_time = self.predictor.predict(self.image)
            save_prediction_results(annotated_img, detections, self.metadata, filename, inference_time)
            self.finished.emit(annotated_img, detections, inference_time, filename)
        except Exception as e:
            logger.error("PredictionWorker error: %s", e)
            self.error.emit(str(e))

# ----------------- UI Pages -----------------
class MainFormPage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Header with logos and title
        header = QHBoxLayout()
        left_logo = QLabel()
        left_logo.setFixedWidth(120)
        left_logo.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        left_logo_path = 'assets/logo_left.png'
        if os.path.exists(left_logo_path):
            left_logo.setPixmap(QPixmap(left_logo_path).scaledToWidth(100, Qt.SmoothTransformation))
        else:
            left_logo.setText("Left Logo")

        title = QLabel("<h1>Frozen Semen Straw Counter</h1>")
        title.setAlignment(Qt.AlignCenter)

        right_logo = QLabel()
        right_logo.setFixedWidth(120)
        right_logo.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        right_logo_path = 'assets/logo_right.png'
        if os.path.exists(right_logo_path):
            right_logo.setPixmap(QPixmap(right_logo_path).scaledToWidth(100, Qt.SmoothTransformation))
        else:
            right_logo.setText("Right Logo")

        header.addWidget(left_logo)
        header.addWidget(title, stretch=1)
        header.addWidget(right_logo)
        layout.addLayout(header)

        layout.addSpacing(10)
        
        # Form group
        form_group = QGroupBox("Bull Information")
        form_layout = QGridLayout()

        # Left column
        self.bull_name = QLineEdit()
        self.reg_no = QLineEdit()
        self.batch_id = QLineEdit()
        self.ai_station = QComboBox()
        self.ai_station.addItems(app_config.get('ai_stations', []))

        form_layout.addWidget(QLabel("Bull Name *"), 0, 0)
        form_layout.addWidget(self.bull_name, 0, 1)
        form_layout.addWidget(QLabel("Registration No. *"), 1, 0)
        form_layout.addWidget(self.reg_no, 1, 1)
        form_layout.addWidget(QLabel("Batch ID"), 2, 0)
        form_layout.addWidget(self.batch_id, 2, 1)
        form_layout.addWidget(QLabel("AI-Station *"), 3, 0)
        form_layout.addWidget(self.ai_station, 3, 1)

        # Right column
        self.destination = QLineEdit()
        self.order_id = QLineEdit()
        self.straw_size = QComboBox()
        self.straw_size.addItems(app_config.get('straw_sizes', []))

        form_layout.addWidget(QLabel("Destination"), 0, 2)
        form_layout.addWidget(self.destination, 0, 3)
        form_layout.addWidget(QLabel("Order ID *"), 1, 2)
        form_layout.addWidget(self.order_id, 1, 3)
        form_layout.addWidget(QLabel("Straw Size *"), 2, 2)
        form_layout.addWidget(self.straw_size, 2, 3)

        form_group.setLayout(form_layout)
        layout.addWidget(form_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.proceed_btn = QPushButton("Proceed to Capture")
        self.proceed_btn.clicked.connect(self.on_proceed)
        self.save_cfg_btn = QPushButton("Save Config")
        self.save_cfg_btn.clicked.connect(self.save_config)

        btn_layout.addWidget(self.proceed_btn)
        btn_layout.addWidget(self.save_cfg_btn)
        layout.addLayout(btn_layout)

        # Footer
        layout.addSpacing(8)
        layout.addWidget(QLabel("Fields marked with * are required. Registration No. must be alphanumeric."))
        layout.addStretch(1)
        self.setLayout(layout)

    def on_proceed(self):
        bname = self.bull_name.text().strip()
        reg = self.reg_no.text().strip()
        order = self.order_id.text().strip()
        if not bname or not reg or not order:
            QMessageBox.warning(self, "Missing fields", "Please fill in all required fields marked with *")
            return
        if not re.match(r'^[a-zA-Z0-9]+$', reg):
            QMessageBox.warning(self, "Invalid Registration No.", "Registration No. must be alphanumeric")
            return
        self.parent.metadata = {
            'bull_name': bname,
            'reg_no': reg,
            'batch_id': self.batch_id.text().strip(),
            'ai_station': self.ai_station.currentText(),
            'destination': self.destination.text().strip(),
            'order_id': order,
            'straw_size': self.straw_size.currentText()
        }
        self.parent.setCurrentIndex(self.parent.index_of_capture)

    def save_config(self):
        save_config(app_config)
        QMessageBox.information(self, "Config Saved", f"Configuration saved to {CONFIG_PATH}")

class CapturePage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.camera_manager = CameraManager(max_indices=app_config.get('camera_check_indices', 10))
        self.preview_timer = QTimer()
        self.preview_timer.setInterval(100)  # 100ms interval to reduce flicker
        self.preview_timer.timeout.connect(self.update_preview)
        self.current_frame = None
        self.captured_frame = None  # Store captured image
        self.connected_camera_info = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Top controls
        top_row = QHBoxLayout()
        back_btn = QPushButton("← Back to Form")
        back_btn.clicked.connect(self.go_back)
        top_row.addWidget(back_btn)
        top_row.addStretch(1)
        layout.addLayout(top_row)

        # Camera / upload mode selection
        mode_box = QGroupBox("Input Method")
        mode_layout = QHBoxLayout()
        self.input_mode_combo = QComboBox()
        self.input_mode_combo.addItems(["Capture from Camera", "Upload Image"])
        self.input_mode_combo.currentIndexChanged.connect(self.on_input_mode_changed)
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.input_mode_combo)
        mode_box.setLayout(mode_layout)
        layout.addWidget(mode_box)

        # Camera selection
        cam_group = QGroupBox("Camera Selection")
        cam_layout = QGridLayout()
        self.cam_list = QComboBox()
        self.refresh_cameras_btn = QPushButton("Refresh Cameras")
        self.refresh_cameras_btn.clicked.connect(self.refresh_camera_list)
        cam_layout.addWidget(QLabel("Available Cameras:"), 0, 0)
        cam_layout.addWidget(self.cam_list, 0, 1)
        cam_layout.addWidget(self.refresh_cameras_btn, 0, 2)

        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("rtsp://username:password@ip:port/stream (optional)")
        cam_layout.addWidget(QLabel("RTSP/URL (optional):"), 1, 0)
        cam_layout.addWidget(self.rtsp_input, 1, 1, 1, 2)

        self.connect_cam_btn = QPushButton("Connect Camera")
        self.connect_cam_btn.clicked.connect(self.connect_camera)
        cam_layout.addWidget(self.connect_cam_btn, 2, 0, 1, 3)

        cam_group.setLayout(cam_layout)
        layout.addWidget(cam_group)

        # Preview area
        preview_group = QGroupBox("Live Preview / Captured Image")
        pv_layout = QVBoxLayout()
        self.preview_label = QLabel("Camera preview will appear here")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(800, 600)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setStyleSheet("background-color: black; border: 2px solid #333;")
        pv_layout.addWidget(self.preview_label)

        ctl_row = QHBoxLayout()
        self.capture_btn = QPushButton("📷 Capture")
        self.capture_btn.clicked.connect(self.capture_image)
        self.retake_btn = QPushButton("🔄 Retake")
        self.retake_btn.clicked.connect(self.retake)
        self.to_prediction_btn = QPushButton("➡️ Proceed to Prediction")
        self.to_prediction_btn.clicked.connect(self.proceed_to_prediction)

        ctl_row.addWidget(self.capture_btn)
        ctl_row.addWidget(self.retake_btn)
        ctl_row.addWidget(self.to_prediction_btn)
        pv_layout.addLayout(ctl_row)
        preview_group.setLayout(pv_layout)
        layout.addWidget(preview_group)

        # Upload area
        upload_group = QGroupBox("Upload / Batch Processing")
        up_layout = QVBoxLayout()
        up_btn_row = QHBoxLayout()
        self.upload_single_btn = QPushButton("Upload Single Image")
        self.upload_single_btn.clicked.connect(self.upload_single_image)
        self.upload_batch_btn = QPushButton("Upload Multiple Images (Batch)")
        self.upload_batch_btn.clicked.connect(self.upload_batch_images)
        up_btn_row.addWidget(self.upload_single_btn)
        up_btn_row.addWidget(self.upload_batch_btn)
        up_layout.addLayout(up_btn_row)

        self.batch_options_cb = QCheckBox("Use same metadata for all images (batch)")
        self.batch_auto_process_cb = QCheckBox("Auto-process batch (run YOLO)")
        up_layout.addWidget(self.batch_options_cb)
        up_layout.addWidget(self.batch_auto_process_cb)

        self.batch_list_widget = QListWidget()
        up_layout.addWidget(self.batch_list_widget)

        upload_group.setLayout(up_layout)
        layout.addWidget(upload_group)

        layout.addStretch(1)
        self.setLayout(layout)

        self.on_input_mode_changed(0)
        self.refresh_camera_list()

    def on_input_mode_changed(self, idx: int):
        mode = self.input_mode_combo.currentText()
        if mode == "Capture from Camera":
            self.cam_list.setEnabled(True)
            self.refresh_cameras_btn.setEnabled(True)
            self.connect_cam_btn.setEnabled(True)
            self.rtsp_input.setEnabled(True)
            self.capture_btn.setEnabled(True)
            self.upload_single_btn.setEnabled(False)
            self.upload_batch_btn.setEnabled(False)
        else:
            self.cam_list.setEnabled(False)
            self.refresh_cameras_btn.setEnabled(False)
            self.connect_cam_btn.setEnabled(False)
            self.rtsp_input.setEnabled(False)
            self.capture_btn.setEnabled(False)
            self.upload_single_btn.setEnabled(True)
            self.upload_batch_btn.setEnabled(True)

    def refresh_camera_list(self):
        self.cam_list.clear()
        cams = self.camera_manager.discover_usb_cameras()
        for cam in cams:
            self.cam_list.addItem(f"{cam['name']} (index {cam['index']})", cam)
        self.cam_list.addItem("RTSP/IP Camera (use URL field)", {'type': 'RTSP', 'name': 'RTSP Camera'})
        logger.info("Camera list refreshed: %d cameras", len(cams))

    def connect_camera(self):
        url = self.rtsp_input.text().strip()
        if url:
            cam_info = {'type': 'RTSP', 'url': url, 'name': 'RTSP Camera'}
        else:
            idx = self.cam_list.currentIndex()
            if idx < 0:
                QMessageBox.warning(self, "No Camera", "Please select a camera or provide RTSP URL")
                return
            cam_info = self.cam_list.itemData(idx)
            if cam_info is None:
                QMessageBox.warning(self, "Invalid Selection", "Please select a valid camera")
                return
        ok = self.camera_manager.open_camera(cam_info)
        if ok:
            self.connected_camera_info = cam_info
            QMessageBox.information(self, "Connected", f"Connected to {cam_info.get('name')}")
            self.captured_frame = None  # Clear any captured frame
            self.preview_timer.start()
        else:
            QMessageBox.critical(self, "Connection Failed", "Unable to open selected camera")

    def update_preview(self):
        # If captured frame exists, show it instead of live preview
        if self.captured_frame is not None:
            self.display_image(self.captured_frame)
            return
            
        frame = self.camera_manager.read_frame()
        if frame is not None:
            self.current_frame = frame
            self.display_image(frame)
        else:
            self.preview_label.setText("No frame available")

    def display_image(self, frame):
        """Display image with proper scaling to prevent flicker"""
        h, w = frame.shape[:2]
        # Scale down for display only
        max_w, max_h = 1280, 720
        scale = min(max_w / w, max_h / h)
        if scale < 1.0:
            frame_display = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            frame_display = frame

        rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(pix.scaled(
            self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def capture_image(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "No Frame", "No camera frame available to capture.")
            return
        
        # Stop live preview and show captured frame
        self.captured_frame = self.current_frame.copy()
        self.preview_timer.stop()
        
        filename = generate_filename(self.parent.metadata)
        save_image_and_metadata(self.captured_frame, self.parent.metadata, 
                               self.connected_camera_info or {'type': 'Camera'}, filename)
        self.parent.captured_image = self.captured_frame.copy()
        self.parent.captured_filename = filename
        
        # Display captured image
        self.display_image(self.captured_frame)
        
        QMessageBox.information(self, "Captured", f"Image captured and saved as {filename}.jpg")

    def retake(self):
        self.captured_frame = None
        self.parent.captured_image = None
        self.parent.captured_filename = None
        QMessageBox.information(self, "Retake", "Ready to capture new image.")
        
        # Resume live preview
        if self.camera_manager.cap and self.camera_manager.cap.isOpened():
            self.preview_timer.start()

    def proceed_to_prediction(self):
        if getattr(self.parent, 'captured_image', None) is None:
            QMessageBox.warning(self, "No Image", "No captured/uploaded image available. Capture or upload first.")
            return
        self.parent.setCurrentIndex(self.parent.index_of_prediction)

    def upload_single_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if not file_path:
            return
        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, "Invalid Image", "Could not read image file.")
            return
        filename = generate_filename(self.parent.metadata)
        camera_info = {'type': 'Upload', 'original_filename': os.path.basename(file_path)}
        save_image_and_metadata(img, self.parent.metadata, camera_info, filename)
        self.parent.captured_image = img
        self.parent.captured_filename = filename
        
        # Display uploaded image
        self.captured_frame = img.copy()
        self.display_image(self.captured_frame)
        
        QMessageBox.information(self, "Uploaded", f"Image uploaded and saved as {filename}.jpg")

    def upload_batch_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select images", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if not file_paths:
            return
        self.batch_list_widget.clear()
        predictor = YOLOPredictor(app_config['model_path'], app_config['confidence_threshold']) if self.batch_auto_process_cb.isChecked() else None
        for i, fp in enumerate(file_paths):
            try:
                img = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    self.batch_list_widget.addItem(f"❌ {os.path.basename(fp)} - failed to decode")
                    continue
                if self.batch_options_cb.isChecked():
                    metadata = self.parent.metadata.copy()
                else:
                    metadata = {
                        'bull_name': f"Batch_{i+1}",
                        'reg_no': f"Batch_{i+1}",
                        'order_id': f"Batch_{i+1}"
                    }
                metadata['batch_index'] = i + 1
                filename = generate_filename(metadata)
                camera_info = {'type': 'Batch Upload', 'original_filename': os.path.basename(fp), 'batch_index': i+1}
                save_image_and_metadata(img, metadata, camera_info, filename)
                if predictor is not None:
                    annotated, detections, inf_t = predictor.predict(img)
                    save_prediction_results(annotated, detections, metadata, filename, inf_t)
                    self.batch_list_widget.addItem(f"✅ {os.path.basename(fp)} -> processed ({len(detections)} dets)")
                else:
                    self.batch_list_widget.addItem(f"✅ {os.path.basename(fp)} -> saved")
            except Exception as e:
                logger.error("Batch upload error for %s: %s", fp, e)
                self.batch_list_widget.addItem(f"❌ {os.path.basename(fp)} -> error: {e}")

    def go_back(self):
        self.preview_timer.stop()
        self.camera_manager.release()
        self.captured_frame = None
        self.parent.setCurrentIndex(self.parent.index_of_main)

    def closeEvent(self, event):
        self.preview_timer.stop()
        self.camera_manager.release()
        super().closeEvent(event)

class PredictionPage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.predictor = YOLOPredictor(app_config['model_path'], app_config['confidence_threshold'])
        self.worker_thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        top_row = QHBoxLayout()
        self.back_btn = QPushButton("← New Capture")
        self.back_btn.clicked.connect(self.go_back)
        top_row.addWidget(self.back_btn)
        top_row.addStretch(1)
        layout.addLayout(top_row)

        self.status_label = QLabel("Ready to run prediction.")
        layout.addWidget(self.status_label)

        # Create horizontal layout for original and predicted images
        images_layout = QHBoxLayout()
        
        # Original Image
        original_group = QGroupBox("Original Image")
        original_layout = QVBoxLayout()
        self.original_image_label = QLabel("Original image will display here")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(400, 400)
        self.original_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.original_image_label.setStyleSheet("background-color: black; border: 2px solid #333;")
        original_layout.addWidget(self.original_image_label)
        original_group.setLayout(original_layout)
        images_layout.addWidget(original_group)
        
        # Predicted Image
        predicted_group = QGroupBox("Predicted Image")
        predicted_layout = QVBoxLayout()
        self.predicted_image_label = QLabel("Predicted image will display here")
        self.predicted_image_label.setAlignment(Qt.AlignCenter)
        self.predicted_image_label.setMinimumSize(400, 400)
        self.predicted_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.predicted_image_label.setStyleSheet("background-color: black; border: 2px solid #333;")
        predicted_layout.addWidget(self.predicted_image_label)
        predicted_group.setLayout(predicted_layout)
        images_layout.addWidget(predicted_group)
        
        layout.addLayout(images_layout)

        self.run_btn = QPushButton("Run YOLO Prediction")
        self.run_btn.clicked.connect(self.run_prediction)
        self.save_btn = QPushButton("Open Results Folder")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)

        hl = QHBoxLayout()
        hl.addWidget(self.run_btn)
        hl.addWidget(self.save_btn)
        layout.addLayout(hl)

        # Detection details
        self.dets_text = QTextEdit()
        self.dets_text.setReadOnly(True)
        self.dets_text.setMaximumHeight(150)
        layout.addWidget(QLabel("Detection Details:"))
        layout.addWidget(self.dets_text)

        self.setLayout(layout)

    def showEvent(self, event):
        super().showEvent(event)
        img = getattr(self.parent, 'captured_image', None)
        if img is None:
            self.original_image_label.setText("No image available. Go back and capture/upload an image first.")
            self.predicted_image_label.setText("No prediction yet")
            self.run_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.status_label.setText("No image available.")
            self.dets_text.clear()
            return
        self.run_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        
        # Display original image
        self.display_image(img, self.original_image_label)
        self.predicted_image_label.setText("Click 'Run YOLO Prediction' to see results")
        self.status_label.setText("Ready to run prediction.")
        self.dets_text.clear()

    def display_image(self, img, label_widget):
        """Display image in the specified label widget with proper scaling"""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        
        # Scale for display
        max_size = min(label_widget.width(), label_widget.height())
        scale = min(max_size / w, max_size / h, 1.0)
        
        if scale < 1.0:
            img_display = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img_display = rgb
        
        h, w = img_display.shape[:2]
        qimg = QImage(img_display.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        label_widget.setPixmap(pix.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def run_prediction(self):
        img = getattr(self.parent, 'captured_image', None)
        if img is None:
            QMessageBox.warning(self, "No Image", "No captured/uploaded image available.")
            return
        
        self.run_btn.setEnabled(False)
        self.status_label.setText("Running YOLO inference...")
        self.dets_text.clear()
        self.predicted_image_label.setText("Processing...")
        
        self.worker_thread = PredictionWorker(self.predictor, img, self.parent.metadata, {'type': 'User'})
        self.worker_thread.finished.connect(self.on_prediction_finished)
        self.worker_thread.error.connect(self.on_prediction_error)
        self.worker_thread.start()

    def on_prediction_finished(self, annotated_img: np.ndarray, detections: list, inference_time: float, filename: str):
        try:
            self.status_label.setText(f"Prediction completed in {inference_time:.2f}s - Saved as: {filename}")
            self.save_btn.setEnabled(True)
            self.run_btn.setEnabled(True)
            
            # Display predicted image
            self.display_image(annotated_img, self.predicted_image_label)
            
            # Populate detections
            txt_lines = []
            for i, d in enumerate(detections, 1):
                cls = d.get('class', 'N/A')
                conf = d.get('confidence', 0.0)
                bbox = d.get('bbox', [])
                txt_lines.append(f"{i}. {cls} - {conf:.2%} - BBox: {bbox}")
            if not txt_lines:
                txt_lines = ["No detections found."]
            self.dets_text.setPlainText("\n".join(txt_lines))
            logger.info("Prediction finished and UI updated.")
        except Exception as e:
            logger.error("on_prediction_finished error: %s", e)
        finally:
            self.worker_thread = None

    def on_prediction_error(self, message: str):
        QMessageBox.critical(self, "Prediction Error", f"Prediction failed: {message}")
        self.status_label.setText("Prediction failed.")
        self.run_btn.setEnabled(True)
        self.predicted_image_label.setText("Prediction failed")
        self.worker_thread = None

    def save_results(self):
        pred_dir = app_config['output_dirs']['predictions']
        QMessageBox.information(self, "Saved", f"Prediction results were saved to {pred_dir}")
        try:
            if sys.platform == "win32":
                os.startfile(pred_dir)
            elif sys.platform == "darwin":
                os.system(f"open {pred_dir}")
            else:
                os.system(f"xdg-open {pred_dir}")
        except Exception as e:
            logger.warning("Could not open prediction directory: %s", e)

    def go_back(self):
        self.parent.captured_image = None
        self.parent.captured_filename = None
        self.parent.setCurrentIndex(self.parent.index_of_main)

# ----------------- Main Application -----------------
class FrozenStrawApp(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.metadata = {}
        self.captured_image = None
        self.captured_filename = None

        self.main_page = MainFormPage(self)
        self.capture_page = CapturePage(self)
        self.prediction_page = PredictionPage(self)

        self.addWidget(self.main_page)
        self.index_of_main = self.indexOf(self.main_page)
        self.addWidget(self.capture_page)
        self.index_of_capture = self.indexOf(self.capture_page)
        self.addWidget(self.prediction_page)
        self.index_of_prediction = self.indexOf(self.prediction_page)

        self.setWindowTitle("Frozen Semen Straw Counter")
        icon_path = 'assets/Everse_logo.png'
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

# ----------------- Run -----------------
def main():
    app = QApplication(sys.argv)
    win = FrozenStrawApp()
    win.showMaximized()  # Open in maximized mode
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
