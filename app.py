import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import threading
from predict import predict_mri, get_model

# Colors
BG_COLOR = "#1e1e2e"
FG_COLOR = "#cdd6f4"
BTN_COLOR = "#89b4fa"
BTN_HOVER = "#b4befe"
ACCENT_COLOR = "#f38ba8"
SUCCESS_COLOR = "#a6e3a1"
WARNING_COLOR = "#f9e2af"
PANEL_BG = "#313244"

class BrainTumorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Detection System")
        self.root.geometry("1000x700")
        self.root.configure(bg=BG_COLOR)
        self.root.minsize(900, 600)
        
        self.current_image_path = None
        
        # UI Setup
        self.setup_ui()
        
        # Try loading model in background to avoid freezing initially
        self.load_model_thread()

    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg=BG_COLOR, pady=20)
        header_frame.pack(fill=tk.X)
        
        title_lbl = tk.Label(header_frame, text="🧠 Brain Tumor Detection System", 
                             font=("Segoe UI", 24, "bold"), bg=BG_COLOR, fg=FG_COLOR)
        title_lbl.pack()
        
        subtitle = tk.Label(header_frame, text="Upload an MRI scan to detect and localize tumors automatically using CNN.", 
                            font=("Segoe UI", 12), bg=BG_COLOR, fg="#a6adc8")
        subtitle.pack()

        # Main Content
        self.content_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=10)
        
        # Split Panels: Images
        self.image_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left Panel (Original)
        self.left_panel = tk.Frame(self.image_frame, bg=PANEL_BG, bd=2, relief=tk.FLAT)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.orig_img_lbl = tk.Label(self.left_panel, text="No Image Uploaded", bg=PANEL_BG, fg="#a6adc8", font=("Segoe UI", 14))
        self.orig_img_lbl.pack(expand=True, fill=tk.BOTH)
        
        orig_title = tk.Label(self.left_panel, text="Original MRI", bg=PANEL_BG, fg=FG_COLOR, font=("Segoe UI", 12, "bold"))
        orig_title.pack(side=tk.BOTTOM, pady=10)

        # Right Panel (Result)
        self.right_panel = tk.Frame(self.image_frame, bg=PANEL_BG, bd=2, relief=tk.FLAT)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.res_img_lbl = tk.Label(self.right_panel, text="Prediction Pending", bg=PANEL_BG, fg="#a6adc8", font=("Segoe UI", 14))
        self.res_img_lbl.pack(expand=True, fill=tk.BOTH)
        
        res_title = tk.Label(self.right_panel, text="Processed / Localization", bg=PANEL_BG, fg=FG_COLOR, font=("Segoe UI", 12, "bold"))
        res_title.pack(side=tk.BOTTOM, pady=10)
        
        # Results Text Display
        self.result_frame = tk.Frame(self.content_frame, bg=BG_COLOR, pady=20)
        self.result_frame.pack(fill=tk.X)
        
        self.prediction_lbl = tk.Label(self.result_frame, text="Awaiting upload...", 
                                       font=("Segoe UI", 18, "bold"), bg=BG_COLOR, fg=WARNING_COLOR)
        self.prediction_lbl.pack()
        
        self.confidence_lbl = tk.Label(self.result_frame, text="", 
                                       font=("Segoe UI", 14), bg=BG_COLOR, fg=FG_COLOR)
        self.confidence_lbl.pack()
        
        # Control Buttons
        control_frame = tk.Frame(self.root, bg=BG_COLOR, pady=20)
        control_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.upload_btn = tk.Button(control_frame, text="Upload MRI Image", font=("Segoe UI", 12, "bold"),
                                    bg=BTN_COLOR, fg="#11111b", activebackground=BTN_HOVER,
                                    padx=20, pady=10, relief=tk.FLAT, cursor="hand2",
                                    command=self.upload_image)
        self.upload_btn.pack(side=tk.LEFT, expand=True)

        self.predict_btn = tk.Button(control_frame, text="Run Prediction", font=("Segoe UI", 12, "bold"),
                                     bg=ACCENT_COLOR, fg="#11111b", activebackground=BTN_HOVER,
                                     padx=20, pady=10, relief=tk.FLAT, cursor="hand2",
                                     command=self.run_prediction_thread, state=tk.DISABLED)
        self.predict_btn.pack(side=tk.RIGHT, expand=True)

        # Loading Progress
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')

    def load_model_thread(self):
        def _load():
            try:
                self.prediction_lbl.config(text="Loading Model...", fg=WARNING_COLOR)
                get_model()
                self.prediction_lbl.config(text="Model Ready. Please upload an image.", fg=SUCCESS_COLOR)
            except Exception as e:
                self.prediction_lbl.config(text="Model not found. Train the model first.", fg=ACCENT_COLOR)
        
        threading.Thread(target=_load, daemon=True).start()

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select MRI Image", 
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        if file_path:
            self.current_image_path = file_path
            self.display_image(self.current_image_path, self.orig_img_lbl)
            self.res_img_lbl.config(image='', text="Prediction Pending")
            self.prediction_lbl.config(text="Ready to Predict", fg=WARNING_COLOR)
            self.confidence_lbl.config(text="")
            self.predict_btn.config(state=tk.NORMAL)

    def display_image(self, img_path_or_array, label_widget):
        try:
            if isinstance(img_path_or_array, str):
                img = Image.open(img_path_or_array)
            else:
                img = Image.fromarray(img_path_or_array)
                
            # Resize for display
            display_size = (350, 350)
            img.thumbnail(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            label_widget.config(image=photo, text="")
            label_widget.image = photo # Keep reference
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def run_prediction_thread(self):
        if not self.current_image_path:
            return
            
        self.predict_btn.config(state=tk.DISABLED)
        self.upload_btn.config(state=tk.DISABLED)
        self.prediction_lbl.config(text="Analyzing...", fg=WARNING_COLOR)
        self.confidence_lbl.config(text="")
        self.progress.pack(fill=tk.X, side=tk.BOTTOM)
        self.progress.start()
        
        # Run prediction in background to keep GUI responsive
        threading.Thread(target=self._predict_process, daemon=True).start()
        
    def _predict_process(self):
        try:
            result = predict_mri(self.current_image_path)
            
            # Must update GUI from main thread
            self.root.after(0, self._update_results, result)
        except Exception as e:
            self.root.after(0, self._show_error, str(e))

    def _update_results(self, result):
        self.progress.stop()
        self.progress.pack_forget()
        self.predict_btn.config(state=tk.NORMAL)
        self.upload_btn.config(state=tk.NORMAL)
        
        pred_class = result['class']
        conf = result['confidence']
        
        if pred_class == 'No Tumor':
            color = SUCCESS_COLOR
            alert_text = "Analysis Complete: No Tumor Detected"
        else:
            color = ACCENT_COLOR
            alert_text = f"Tumor Detected: {pred_class}"
            
        self.prediction_lbl.config(text=alert_text, fg=color)
        self.confidence_lbl.config(text=f"Confidence: {conf:.2f}%", fg=FG_COLOR)
        
        # Display Marked Image
        self.display_image(result['marked_img'], self.res_img_lbl)
        
    def _show_error(self, error_msg):
        self.progress.stop()
        self.progress.pack_forget()
        self.predict_btn.config(state=tk.NORMAL)
        self.upload_btn.config(state=tk.NORMAL)
        self.prediction_lbl.config(text="An error occurred", fg=ACCENT_COLOR)
        messagebox.showerror("Prediction Error", error_msg)

if __name__ == "__main__":
    from sys import platform
    
    # Enable High DPI support on Windows
    if platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    root = tk.Tk()
    app = BrainTumorApp(root)
    root.mainloop()
