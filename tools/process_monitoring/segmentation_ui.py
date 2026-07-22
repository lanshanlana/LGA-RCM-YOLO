import sys
import os
import argparse
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Try to import OpenCV for BGR->RGB conversion
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    def cvtColor_bgr2rgb(image):
        return image[..., ::-1]


def run_headless(model_path, input_path, output_dir):
    """
    Headless batch processing mode.
    Processes all images in a folder or a single image, saves results to output_dir.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine input files
    image_files = []
    if os.path.isdir(input_path):
        for f in os.listdir(input_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_files.append(os.path.join(input_path, f))
        image_files.sort()
    elif os.path.isfile(input_path):
        image_files = [input_path]
    else:
        print(f"Error: Input path '{input_path}' is not a valid file or directory.", file=sys.stderr)
        sys.exit(1)

    if not image_files:
        print("No supported image files found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(image_files)} image(s). Processing...")

    for idx, img_path in enumerate(image_files):
        print(f"[{idx+1}/{len(image_files)}] Processing {os.path.basename(img_path)}...")
        try:
            results = model(img_path)
            if results:
                # Plot result (BGR numpy array)
                result_img = results[0].plot()
                # Convert BGR to RGB for correct color saving
                if CV2_AVAILABLE:
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                else:
                    result_img_rgb = cvtColor_bgr2rgb(result_img)
                
                # Save using PIL
                out_path = os.path.join(output_dir, f"seg_{os.path.basename(img_path)}")
                Image.fromarray(result_img_rgb).save(out_path)
                print(f"  Saved to {out_path}")
            else:
                print(f"  Warning: No results for {img_path}")
        except Exception as e:
            print(f"  Error processing {img_path}: {e}", file=sys.stderr)

    print(f"All done. Results saved to {output_dir}")


# ==================== GUI CLASS (only used in non-headless mode) ====================
# Import Qt only when needed to avoid dependency issues in headless mode
if not ('--headless' in sys.argv or '-h' in sys.argv or '--help' in sys.argv):
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                                 QPushButton, QLabel, QFileDialog, QSizePolicy, QScrollArea, QFrame)
    from PyQt5.QtGui import QPixmap, QImage
    from PyQt5.QtCore import Qt

    class YOLOSegmentationGUI(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Experimental Instrument Segmentation System")
            self.setGeometry(100, 100, 1000, 700)

            self.central_widget = QWidget()
            self.setCentralWidget(self.central_widget)
            self.main_layout = QHBoxLayout(self.central_widget)

            # --- Left: Image display ---
            self.image_label = QLabel("Please select a model and an image")
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.image_label.setScaledContents(False)

            self.scroll_area = QScrollArea()
            self.scroll_area.setWidgetResizable(True)
            self.scroll_area.setWidget(self.image_label)
            self.main_layout.addWidget(self.scroll_area, 2)

            # --- Right: Control panel ---
            self.control_layout = QVBoxLayout()

            self.btn_select_model = QPushButton("Select Model (.pt)")
            self.btn_select_model.clicked.connect(self.select_model)
            self.lbl_model_path = QLabel("Model Path: Not selected")
            self.lbl_model_path.setWordWrap(True)

            self.btn_select_folder = QPushButton("Select Image Folder")
            self.btn_select_folder.clicked.connect(self.select_folder)
            self.lbl_folder_path = QLabel("Image Folder: Not selected")
            self.lbl_folder_path.setWordWrap(True)

            self.btn_select_single_image = QPushButton("Select Single Image")
            self.btn_select_single_image.clicked.connect(self.select_single_image)
            self.lbl_current_image = QLabel("Current Image: Not selected")
            self.lbl_current_image.setWordWrap(True)

            self.btn_previous = QPushButton("Previous")
            self.btn_previous.clicked.connect(self.load_previous_image)
            self.btn_previous.setEnabled(False)

            self.btn_next = QPushButton("Next")
            self.btn_next.clicked.connect(self.load_next_image)
            self.btn_next.setEnabled(False)

            self.btn_run_inference = QPushButton("Run Segmentation")
            self.btn_run_inference.clicked.connect(self.run_inference)
            self.btn_run_inference.setEnabled(False)

            self.control_layout.addWidget(self.btn_select_model)
            self.control_layout.addWidget(self.lbl_model_path)
            self._add_separator()
            self.control_layout.addWidget(self.btn_select_folder)
            self.control_layout.addWidget(self.lbl_folder_path)
            self.control_layout.addWidget(self.btn_select_single_image)
            self.control_layout.addWidget(self.lbl_current_image)
            self._add_separator()
            self.control_layout.addWidget(self.btn_previous)
            self.control_layout.addWidget(self.btn_next)
            self._add_separator()
            self.control_layout.addWidget(self.btn_run_inference)

            self.control_layout.addStretch(1)
            self.main_layout.addLayout(self.control_layout, 1)

            # State
            self.model = None
            self.image_folder = None
            self.image_files = []
            self.current_image_index = -1
            self.current_image_path = None

        def _add_separator(self):
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setFrameShadow(QFrame.Sunken)
            self.control_layout.addWidget(sep)

        def load_model(self, model_path):
            if not os.path.exists(model_path):
                self.lbl_model_path.setText(f"Model Path: File not found - {model_path}")
                return False
            try:
                self.model = YOLO(model_path)
                self.lbl_model_path.setText(f"Model Path: {model_path}")
                self.btn_run_inference.setEnabled(self.current_image_path is not None)
                print(f"Model loaded successfully: {model_path}")
                return True
            except Exception as e:
                self.model = None
                self.lbl_model_path.setText(f"Model loading failed: {str(e)}")
                self.btn_run_inference.setEnabled(False)
                print(f"Model loading failed: {e}", file=sys.stderr)
                return False

        def select_model(self):
            path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "PyTorch Models (*.pt);;All Files (*)")
            if path:
                self.load_model(path)

        def select_folder(self):
            folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
            if folder:
                self.image_folder = folder
                self.lbl_folder_path.setText(f"Image Folder: {folder}")
                self.image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff'))])
                self.current_image_index = -1
                self.current_image_path = None
                self.lbl_current_image.setText("Current Image: Not selected")
                self.display_image(None)
                if self.image_files:
                    self.current_image_index = 0
                    self.load_current_image()
                    self.btn_previous.setEnabled(False)
                    self.btn_next.setEnabled(len(self.image_files) > 1)
                else:
                    self.btn_previous.setEnabled(False)
                    self.btn_next.setEnabled(False)
                    print(f"Warning: No supported images in {folder}", file=sys.stderr)

        def select_single_image(self, path=None):
            if path is None:
                path, _ = QFileDialog.getOpenFileName(self, "Select Single Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)")
            if not path:
                return
            self.image_folder = None
            self.image_files = [os.path.basename(path)]
            self.current_image_index = 0
            self.current_image_path = path
            self.lbl_folder_path.setText("Image Folder: Single file mode")
            self.lbl_current_image.setText(f"Current Image: {os.path.basename(path)}")
            self.load_current_image()
            self.btn_previous.setEnabled(False)
            self.btn_next.setEnabled(False)

        def load_current_image(self):
            if self.image_folder and 0 <= self.current_image_index < len(self.image_files):
                self.current_image_path = os.path.join(self.image_folder, self.image_files[self.current_image_index])
                self.lbl_current_image.setText(f"Current Image: {self.image_files[self.current_image_index]}")
                self.display_image(self.current_image_path)
                self.btn_previous.setEnabled(self.current_image_index > 0)
                self.btn_next.setEnabled(self.current_image_index < len(self.image_files) - 1)
            elif self.current_image_path and self.image_folder is None:
                self.display_image(self.current_image_path)
                self.btn_previous.setEnabled(False)
                self.btn_next.setEnabled(False)
            else:
                self.display_image(None)
                self.lbl_current_image.setText("Current Image: Not selected")
                self.btn_previous.setEnabled(False)
                self.btn_next.setEnabled(False)
                self.btn_run_inference.setEnabled(False)
                return
            self.btn_run_inference.setEnabled(self.model is not None)

        def load_previous_image(self):
            if self.image_folder and self.current_image_index > 0:
                self.current_image_index -= 1
                self.load_current_image()

        def load_next_image(self):
            if self.image_folder and self.current_image_index < len(self.image_files) - 1:
                self.current_image_index += 1
                self.load_current_image()

        def display_image(self, image_path):
            if image_path:
                try:
                    pil_img = Image.open(image_path)
                    w, h = pil_img.size
                    if pil_img.mode == 'RGB':
                        fmt, data, bpl = QImage.Format_RGB888, pil_img.tobytes(), w*3
                    elif pil_img.mode == 'L':
                        fmt, data, bpl = QImage.Format_Grayscale8, pil_img.tobytes(), w
                    elif pil_img.mode == 'RGBA':
                        pil_img = pil_img.convert('RGB')
                        fmt, data, bpl = QImage.Format_RGB888, pil_img.tobytes(), w*3
                    else:
                        pil_img = pil_img.convert('RGB')
                        fmt, data, bpl = QImage.Format_RGB888, pil_img.tobytes(), w*3
                    qimg = QImage(data, w, h, bpl, fmt)
                    self.image_label.setPixmap(QPixmap.fromImage(qimg))
                    self.image_label.adjustSize()
                    print(f"Displayed: {os.path.basename(image_path)}")
                except Exception as e:
                    self.image_label.setText(f"Failed to load:\n{os.path.basename(image_path)}\n{e}")
                    self.image_label.setPixmap(QPixmap())
                    self.image_label.adjustSize()
                    print(f"Load error: {e}", file=sys.stderr)
            else:
                self.image_label.setText("Please select a model and an image")
                self.image_label.setPixmap(QPixmap())
                self.image_label.adjustSize()

        def display_result_image(self, np_img):
            if np_img is not None:
                try:
                    if CV2_AVAILABLE:
                        rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                    else:
                        rgb = cvtColor_bgr2rgb(np_img)
                    h, w, ch = rgb.shape
                    qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
                    self.image_label.setPixmap(QPixmap.fromImage(qimg))
                    self.image_label.adjustSize()
                    print("Displayed result")
                except Exception as e:
                    self.image_label.setText(f"Failed to display result:\n{e}")
                    self.image_label.setPixmap(QPixmap())
                    self.image_label.adjustSize()
                    print(f"Result display error: {e}", file=sys.stderr)
            else:
                self.image_label.setText("No result")
                self.image_label.setPixmap(QPixmap())
                self.image_label.adjustSize()

        def run_inference(self):
            if self.model is None:
                print("Error: No model loaded.", file=sys.stderr)
                return
            if not self.current_image_path or not os.path.exists(self.current_image_path):
                print("Error: No valid image.", file=sys.stderr)
                return
            try:
                print(f"Running inference on {os.path.basename(self.current_image_path)}...")
                results = self.model(self.current_image_path)
                if results:
                    self.display_result_image(results[0].plot())
                    print("Inference done.")
                else:
                    print("No results.", file=sys.stderr)
                    self.display_image(self.current_image_path)
            except Exception as e:
                print(f"Inference error: {e}", file=sys.stderr)
                self.display_image(self.current_image_path)
                self.image_label.setText(f"Inference failed:\n{e}")


# ==================== MAIN ENTRY ====================
def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO Segmentation Tool (GUI + Headless)")
    parser.add_argument("--model", type=str, help="Path to YOLO model (.pt)")
    parser.add_argument("--image", type=str, help="Path to a single image file")
    parser.add_argument("--folder", type=str, help="Path to a folder containing images")
    parser.add_argument("--headless", action="store_true", help="Run in headless batch mode (no GUI, saves results)")
    parser.add_argument("--output", type=str, default="./results", help="Output directory for headless mode")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # If headless flag is set, run batch processing and exit
    if args.headless:
        if not args.model:
            print("Error: --model is required in headless mode.", file=sys.stderr)
            sys.exit(1)
        if not args.image and not args.folder:
            print("Error: --image or --folder is required in headless mode.", file=sys.stderr)
            sys.exit(1)
        input_path = args.image if args.image else args.folder
        run_headless(args.model, input_path, args.output)
        sys.exit(0)

    # Check for DISPLAY variable (typical GUI environment check)
    if 'DISPLAY' not in os.environ:
        print(
            "WARNING: DISPLAY environment variable not set. "
            "This likely means no GUI is available. "
            "If you intend to run on a server, please use the --headless flag.",
            file=sys.stderr
        )
        # We don't force exit here, but user will see Qt error if they proceed.

    # Otherwise, start the GUI (with optional pre-loaded args)
    # Lazy import to ensure headless mode doesn't need Qt
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = YOLOSegmentationGUI()

    # Pre-load model if provided via CLI
    if args.model:
        if not window.load_model(args.model):
            print(f"Failed to load model from '{args.model}'. Exiting.", file=sys.stderr)
            sys.exit(1)

    # Pre-select image/folder if provided
    if args.image:
        window.select_single_image(args.image)
    elif args.folder:
        if os.path.isdir(args.folder):
            window.image_folder = args.folder
            window.lbl_folder_path.setText(f"Image Folder: {args.folder}")
            window.image_files = sorted([f for f in os.listdir(args.folder) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff'))])
            if window.image_files:
                window.current_image_index = 0
                window.load_current_image()
                window.btn_previous.setEnabled(False)
                window.btn_next.setEnabled(len(window.image_files) > 1)
            else:
                print(f"Warning: No images in {args.folder}", file=sys.stderr)
        else:
            print(f"Error: Folder '{args.folder}' not found.", file=sys.stderr)

    window.show()
    sys.exit(app.exec_())