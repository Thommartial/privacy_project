#!/usr/bin/env python3
"""
RedactorAppDemo - LARGE TEXT VERSION with Model-Based PII Detection
A modern GUI application for privacy-preserving PII redaction
using ε=8.0 model for PII detection - NO COLOR VERSION.
"""

import sys
import os
from pathlib import Path
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# GUI imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, font
import tkinter.font as tkfont
from PIL import Image, ImageTk
import threading
import queue

# ML imports
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random
import warnings
warnings.filterwarnings('ignore')

# Define the model architecture
class DistilBERTDPModel(nn.Module):
    """DistilBERT-based DP model for PII detection."""
    hidden_size: int = 768
    num_classes: int = 2
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.1)(x, deterministic=True)
        x = nn.Dense(self.hidden_size // 2)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x

class ModelBasedPIIProcessor:
    """Handles PII detection and processing using ε=8.0 model predictions."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.model = None
        self.params = None
        self.config = None
        self.regex_patterns = self._compile_regex_patterns()
        self.load_model()
        
    def _compile_regex_patterns(self):
        """Compile regex patterns for text segmentation."""
        return {
            # For text segmentation
            'word': r'\b\w+\b',
            'punctuation': r'[^\w\s]',
            'whitespace': r'\s+'
        }
    
    def load_model(self):
        """Load the ε=8.0 model for PII detection."""
        try:
            model_path = self.models_dir / "final_epsilon_8.0"
            if model_path.exists():
                # Load model config
                with open(model_path / "config.json", 'r') as f:
                    self.config = json.load(f)
                
                # Load model parameters
                with open(model_path / "best_params.pkl", 'rb') as f:
                    self.params = pickle.load(f)
                
                # Create model instance
                self.model = DistilBERTDPModel(hidden_size=self.config.get('hidden_size', 768))
                print(f"✅ Loaded ε=8.0 model successfully")
                return True
            else:
                print(f"❌ Model not found: {model_path}")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def extract_features(self, text: str) -> List[Dict]:
        """Extract features from text for model prediction."""
        features = []
        words = re.findall(self.regex_patterns['word'], text)
        
        for i, word in enumerate(words):
            # Simple feature extraction (matching training)
            word_features = [
                min(len(word) / 10, 2.0),                    # Length feature
                1.0 if '@' in word else 0.0,                 # Has @
                1.0 if '.' in word and '@' in word else 0.0, # Has . after @
                sum(c.isdigit() for c in word) / max(len(word), 1),  # Digit ratio
                1.0 if word and word[0].isupper() else 0.0,  # Has caps
                len(word.split()) / 10.0                     # Word count
            ]
            
            features.append({
                'word': word,
                'features': np.array(word_features, dtype=np.float32),
                'position': i
            })
        
        return features
    
    def predict_pii(self, text: str) -> List[Dict]:
        """Use ε=8.0 model to predict PII in text."""
        if not self.model or not self.params:
            print("❌ Model not loaded")
            return []
        
        try:
            # Extract features
            features_list = self.extract_features(text)
            if not features_list:
                return []
            
            # Prepare batch for prediction
            feature_vectors = np.array([f['features'] for f in features_list])
            
            # Define prediction function
            def predict_batch(params, batch):
                logits = self.model.apply({'params': params}, batch)
                probs = jax.nn.softmax(logits, axis=-1)
                return probs
            
            # Make predictions
            probs = predict_batch(self.params, feature_vectors)
            probs_np = np.array(probs)
            
            # Convert predictions to PII entities
            pii_entities = []
            for i, (feature, prob) in enumerate(zip(features_list, probs_np)):
                pii_prob = prob[1]  # Probability of being PII
                if pii_prob > 0.5:  # Threshold for PII detection
                    # Find word position in original text
                    word = feature['word']
                    pattern = re.compile(r'\b' + re.escape(word) + r'\b')
                    matches = list(pattern.finditer(text))
                    
                    if matches:
                        match = matches[0]
                        # Determine PII type based on features
                        pii_type = self._classify_pii_type(feature['features'], pii_prob)
                        
                        pii_entities.append({
                            'type': pii_type,
                            'text': word,
                            'start': match.start(),
                            'end': match.end(),
                            'confidence': float(pii_prob)
                        })
            
            return pii_entities
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return []
    
    def _classify_pii_type(self, features: np.ndarray, confidence: float) -> str:
        """Classify PII type based on features."""
        # Extract feature values
        length, has_at, has_dot_after_at, digit_ratio, has_caps, _ = features
        
        # Rule-based classification (simplified)
        if has_at > 0.5 and has_dot_after_at > 0.5:
            return "EMAIL"
        elif digit_ratio > 0.7 and length > 1.0:
            if digit_ratio > 0.9:
                return "CREDIT_CARD"
            else:
                return "PHONE"
        elif has_caps > 0.5 and length > 1.0:
            if digit_ratio < 0.1:
                return "NAME"
            else:
                return "ACCOUNT_NUMBER"
        elif digit_ratio > 0.3:
            return "NUMERIC_ID"
        elif has_at > 0.5:
            return "USERNAME"
        else:
            return "GENERAL_PII"
    
    def detect_pii_with_regex(self, text: str) -> List[Dict]:
        """Fallback: Detect PII using regex patterns."""
        pii_patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            'CREDIT_CARD': r'\b(?:\d{4}[- ]?){3}\d{4}\b',
            'NAME': r'\b(?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.)?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'
        }
        
        pii_entities = []
        for pii_type, pattern in pii_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Simple confidence based on pattern match
                confidence = 0.7  # Base confidence for regex matches
                
                pii_entities.append({
                    'type': pii_type,
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': confidence
                })
        
        return pii_entities
    
    def detect_pii(self, text: str, use_model: bool = True) -> List[Dict]:
        """Detect PII in text using model or regex fallback."""
        if use_model and self.model:
            pii_entities = self.predict_pii(text)
            if pii_entities:
                return pii_entities
        
        # Fallback to regex if model fails or is not available
        return self.detect_pii_with_regex(text)
    
    def redact_text(self, text: str, pii_entities: List[Dict], privacy_level: float = 1.0) -> str:
        """
        Redact text based on PII entities and privacy level.
        Higher privacy = more redaction.
        """
        if not pii_entities:
            return text
        
        # Convert text to list for manipulation
        chars = list(text)
        
        # Calculate redaction probability based on privacy level
        # Lower epsilon (more private) = higher redaction probability
        redaction_prob = min(0.95, max(0.1, 1.0 / privacy_level))
        
        for entity in pii_entities:
            # Adjust redaction probability based on confidence
            confidence_adjusted_prob = redaction_prob * entity['confidence']
            
            if np.random.random() < confidence_adjusted_prob:
                # Redact this entity
                start, end = entity['start'], entity['end']
                redaction_char = '█'  # Full block character
                
                # Replace with redaction characters
                for i in range(start, end):
                    if i < len(chars):
                        chars[i] = redaction_char
        
        return ''.join(chars)

class RedactorAppDemoLarge:
    def __init__(self, root):
        self.root = root
        self.root.title("RedactorAppDemo - Model-Based PII Detection (ε=8.0)")
        
        # Get screen dimensions and set larger window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Use 90% of screen for better visibility
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        self.root.geometry(f"{window_width}x{window_height}")
        
        self.root.configure(bg="#1e272e")
        
        # Configuration
        self.models_dir = Path("/home/thom/Desktop/dpjax/outputs/models")
        self.available_epsilons = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
        self.current_epsilon = 8.0  # Default (least private)
        self.loaded_models = {}
        self.current_file = None
        self.original_text = ""
        self.pii_entities = []
        self.redacted_text = ""
        self.use_model_detection = True
        
        # Font size configuration (TRIPLED from standard)
        self.base_font_size = 16
        self.title_font_size = 36
        self.button_font_size = 14
        self.label_font_size = 14
        self.text_font_size = 18
        
        # Initialize PII processor with model
        self.pii_processor = ModelBasedPIIProcessor(self.models_dir)
        
        # Initialize UI
        self.setup_styles()
        self.create_widgets()
        self.load_models_async()
        
        # Status variables
        self.processing_queue = queue.Queue()
        self.check_queue()
    
    def set_window_icon(self):
        """Set window icon with privacy theme."""
        try:
            from PIL import Image, ImageDraw
            icon = Image.new('RGBA', (96, 96), color=(0, 0, 0, 0))
            draw = ImageDraw.Draw(icon)
            
            draw.ellipse([12, 12, 84, 84], fill='#3498db')
            draw.rectangle([36, 45, 60, 66], fill='#2c3e50')
            draw.rectangle([45, 36, 51, 45], fill='#2c3e50')
            draw.ellipse([42, 27, 54, 39], fill='#2c3e50')
            
            self.icon = ImageTk.PhotoImage(icon)
            self.root.iconphoto(True, self.icon)
        except:
            pass
    
    def setup_styles(self):
        """Configure modern ttk styles with large fonts."""
        style = ttk.Style()
        
        style.theme_create('dark_large', parent='clam', settings={
            'TFrame': {'configure': {'background': '#1e272e'}},
            'TLabel': {
                'configure': {
                    'background': '#1e272e',
                    'foreground': '#ecf0f1',
                    'font': ('Segoe UI', self.label_font_size)
                }
            },
            'TButton': {
                'configure': {
                    'background': '#3498db',
                    'foreground': 'white',
                    'borderwidth': 0,
                    'focuscolor': 'none',
                    'font': ('Segoe UI', self.button_font_size, 'bold'),
                    'padding': (25, 15)
                },
                'map': {
                    'background': [('active', '#2980b9'), ('disabled', '#7f8c8d')],
                    'foreground': [('disabled', '#bdc3c7')]
                }
            },
            'TCombobox': {
                'configure': {
                    'fieldbackground': '#2c3e50',
                    'background': '#2c3e50',
                    'foreground': '#ecf0f1',
                    'arrowcolor': '#ecf0f1',
                    'selectbackground': '#3498db',
                    'selectforeground': 'white',
                    'font': ('Segoe UI', self.label_font_size),
                    'padding': (10, 5)
                }
            },
            'TCheckbutton': {
                'configure': {
                    'background': '#1e272e',
                    'foreground': '#ecf0f1',
                    'font': ('Segoe UI', self.label_font_size)
                }
            }
        })
        style.theme_use('dark_large')
    
    def create_widgets(self):
        """Create all GUI widgets with large text and elements."""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=25, pady=25)
        
        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill='x', pady=(0, 30))
        
        # Title
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side='left', fill='both', expand=True)
        
        title_label = tk.Label(title_frame,
                              text="🔒 REDACTORAPP DEMO",
                              font=('Segoe UI', self.title_font_size, 'bold'),
                              bg='#1e272e',
                              fg='#3498db')
        title_label.pack(side='left')
        
        subtitle_label = tk.Label(title_frame,
                                 text="Model-Based PII Detection (ε=8.0)",
                                 font=('Segoe UI', int(self.title_font_size * 0.6)),
                                 bg='#1e272e',
                                 fg='#7f8c8d')
        subtitle_label.pack(side='left', padx=(20, 0))
        
        # Status indicator
        self.status_frame = ttk.Frame(header_frame)
        self.status_frame.pack(side='right')
        
        self.status_dot = tk.Canvas(self.status_frame, width=40, height=40,
                                   bg='#1e272e', highlightthickness=0)
        self.status_dot.pack(side='left', padx=(0, 20))
        self.draw_status_dot('green')
        
        self.status_label = ttk.Label(self.status_frame, text="READY", 
                                     font=('Segoe UI', self.label_font_size, 'bold'))
        self.status_label.pack(side='left')
        
        # Main content area
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill='both', expand=True)
        
        # Left panel - Original text
        left_panel = ttk.LabelFrame(content_frame, text="📄 ORIGINAL DOCUMENT", 
                                   padding=(25, 20))
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        # File controls
        file_controls = ttk.Frame(left_panel)
        file_controls.pack(fill='x', pady=(0, 20))
        
        self.browse_btn = ttk.Button(file_controls, 
                                    text="📁 BROWSE FILES",
                                    command=self.browse_file,
                                    style='TButton')
        self.browse_btn.pack(side='left', padx=(0, 20))
        
        self.upload_btn = ttk.Button(file_controls,
                                    text="⬆️ UPLOAD & ANALYZE",
                                    command=self.upload_file,
                                    style='TButton')
        self.upload_btn.pack(side='left')
        
        # Detection mode toggle
        self.use_model_var = tk.BooleanVar(value=True)
        self.model_toggle = ttk.Checkbutton(file_controls,
                                          text="Use ε=8.0 Model",
                                          variable=self.use_model_var,
                                          command=self.toggle_detection_mode)
        self.model_toggle.pack(side='left', padx=(20, 0))
        
        self.file_info_label = ttk.Label(file_controls,
                                        text="No file selected",
                                        font=('Segoe UI', self.label_font_size))
        self.file_info_label.pack(side='right')
        
        # Original text display
        text_frame = ttk.Frame(left_panel)
        text_frame.pack(fill='both', expand=True)
        
        text_container = ttk.Frame(text_frame)
        text_container.pack(fill='both', expand=True)
        
        text_scrollbar = ttk.Scrollbar(text_container)
        text_scrollbar.pack(side='right', fill='y')
        
        self.original_text_widget = tk.Text(text_container,
                                           wrap='word',
                                           font=('Segoe UI', self.text_font_size),
                                           bg='#2c3e50',
                                           fg='#ecf0f1',
                                           insertbackground='#ecf0f1',
                                           selectbackground='#3498db',
                                           borderwidth=0,
                                           relief='flat',
                                           padx=25,
                                           pady=25,
                                           yscrollcommand=text_scrollbar.set)
        self.original_text_widget.pack(side='left', fill='both', expand=True)
        text_scrollbar.config(command=self.original_text_widget.yview)
        
        # Right panel - Redacted text
        right_panel = ttk.LabelFrame(content_frame, text="🔒 REDACTED DOCUMENT",
                                    padding=(25, 20))
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Privacy controls
        privacy_frame = ttk.Frame(right_panel)
        privacy_frame.pack(fill='x', pady=(0, 20))
        
        ttk.Label(privacy_frame, 
                 text="PRIVACY LEVEL (ε):",
                 font=('Segoe UI', self.button_font_size, 'bold')).pack(side='left')
        
        self.privacy_var = tk.StringVar(value="8.0")
        self.privacy_combo = ttk.Combobox(privacy_frame,
                                         textvariable=self.privacy_var,
                                         values=[str(e) for e in self.available_epsilons],
                                         state='readonly',
                                         width=15,
                                         font=('Segoe UI', self.label_font_size))
        self.privacy_combo.pack(side='left', padx=(20, 30))
        self.privacy_combo.bind('<<ComboboxSelected>>', self.on_privacy_change)
        
        self.redact_btn = ttk.Button(privacy_frame,
                                    text="🔐 APPLY REDACTION",
                                    command=self.redact_text,
                                    style='TButton')
        self.redact_btn.pack(side='right')
        
        # Redacted text display
        redacted_text_frame = ttk.Frame(right_panel)
        redacted_text_frame.pack(fill='both', expand=True)
        
        redacted_scrollbar = ttk.Scrollbar(redacted_text_frame)
        redacted_scrollbar.pack(side='right', fill='y')
        
        self.redacted_text_widget = tk.Text(redacted_text_frame,
                                           wrap='word',
                                           font=('Segoe UI', self.text_font_size),
                                           bg='#2c3e50',
                                           fg='#ecf0f1',
                                           insertbackground='#ecf0f1',
                                           selectbackground='#3498db',
                                           borderwidth=0,
                                           relief='flat',
                                           padx=25,
                                           pady=25,
                                           yscrollcommand=redacted_scrollbar.set)
        self.redacted_text_widget.pack(side='left', fill='both', expand=True)
        redacted_scrollbar.config(command=self.redacted_text_widget.yview)
        
        # Bottom panel
        bottom_panel = ttk.Frame(main_container)
        bottom_panel.pack(fill='x', pady=(30, 0))
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(bottom_panel, text="📊 DETECTION STATISTICS",
                                    padding=(30, 20))
        stats_frame.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        self.stats_grid = ttk.Frame(stats_frame)
        self.stats_grid.pack(fill='both', expand=True)
        
        self.init_stats_labels()
        
        # Controls frame
        controls_frame = ttk.LabelFrame(bottom_panel, text="⚙️ CONTROLS",
                                       padding=(30, 20))
        controls_frame.pack(side='right', fill='x')
        
        control_buttons = ttk.Frame(controls_frame)
        control_buttons.pack()
        
        self.save_btn = ttk.Button(control_buttons,
                                  text="💾 SAVE REDACTED",
                                  command=self.save_redacted,
                                  style='TButton')
        self.save_btn.pack(side='left', padx=(0, 20))
        
        self.clear_btn = ttk.Button(control_buttons,
                                   text="🗑️ CLEAR ALL",
                                   command=self.clear_all,
                                   style='TButton')
        self.clear_btn.pack(side='left')
    
    def init_stats_labels(self):
        """Initialize statistics labels."""
        stats = [
            ("DOCUMENT SIZE:", "0 chars"),
            ("PII DETECTED:", "0"),
            ("AVG CONFIDENCE:", "0%"),
            ("DETECTION MODE:", "ε=8.0 Model"),
            ("PRIVACY LEVEL:", f"ε={self.current_epsilon}"),
            ("MODEL STATUS:", "Loading..."),
            ("PROCESSING TIME:", "0.00s"),
            ("REDACTION RATE:", "0%")
        ]
        
        for i, (label, value) in enumerate(stats):
            row = i // 2
            col = (i % 2) * 2
            
            label_widget = ttk.Label(self.stats_grid, text=label,
                                    font=('Segoe UI', self.label_font_size),
                                    foreground='#95a5a6')
            label_widget.grid(row=row, column=col, sticky='w', padx=(0, 10), pady=5)
            
            value_widget = ttk.Label(self.stats_grid, text=value,
                                    font=('Segoe UI', self.label_font_size, 'bold'))
            value_widget.grid(row=row, column=col+1, sticky='w', padx=(0, 30), pady=5)
            
            var_name = f'stat_{label.lower().replace(":", "").replace(" ", "_")}'
            setattr(self, var_name, value_widget)
    
    def draw_status_dot(self, color):
        """Draw status indicator dot."""
        self.status_dot.delete('all')
        self.status_dot.create_oval(10, 10, 30, 30, fill=color, outline='', width=2)
    
    def load_models_async(self):
        """Load models in background thread."""
        self.set_status("yellow", "LOADING MODEL...")
        
        def load_task():
            try:
                # Load ε=8.0 model
                model_path = self.models_dir / "final_epsilon_8.0"
                if model_path.exists():
                    with open(model_path / "config.json", 'r') as f:
                        config = json.load(f)
                    
                    with open(model_path / "best_params.pkl", 'rb') as f:
                        params = pickle.load(f)
                    
                    self.loaded_models[8.0] = {
                        'config': config,
                        'params': params,
                        'model': DistilBERTDPModel(hidden_size=config.get('hidden_size', 768))
                    }
                    
                    self.processing_queue.put(('model_loaded', 8.0))
                    self.stat_model_status.config(text="ε=8.0 Model Ready")
                else:
                    self.stat_model_status.config(text="Model Not Found")
                
                self.processing_queue.put(('models_complete', None))
                
            except Exception as e:
                print(f"Error loading model: {e}")
                self.stat_model_status.config(text=f"Error: {str(e)[:30]}")
                self.processing_queue.put(('models_complete', None))
        
        thread = threading.Thread(target=load_task, daemon=True)
        thread.start()
    
    def check_queue(self):
        """Check processing queue for updates."""
        try:
            while True:
                msg_type, data = self.processing_queue.get_nowait()
                
                if msg_type == 'model_loaded':
                    self.set_status("green", "MODEL READY")
                    self.stat_detection_mode.config(text="ε=8.0 Model")
                elif msg_type == 'models_complete':
                    self.set_status("green", "READY")
                elif msg_type == 'processing_start':
                    self.set_status("yellow", "PROCESSING...")
                elif msg_type == 'processing_complete':
                    self.set_status("green", "READY")
                
        except queue.Empty:
            pass
        
        self.root.after(100, self.check_queue)
    
    def set_status(self, color, text):
        """Update status indicator."""
        self.draw_status_dot(color)
        self.status_label.config(text=text)
    
    def toggle_detection_mode(self):
        """Toggle between model and regex detection."""
        self.use_model_detection = self.use_model_var.get()
        mode_text = "ε=8.0 Model" if self.use_model_detection else "Regex Fallback"
        self.stat_detection_mode.config(text=mode_text)
        self.status_label.config(text=f"Detection Mode: {mode_text}")
    
    def browse_file(self):
        """Open file dialog to select text file."""
        file_path = filedialog.askopenfilename(
            title="Select Text File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_file = Path(file_path)
            file_size = self.current_file.stat().st_size
            size_text = f"{file_size/1000:.1f} KB" if file_size < 1000000 else f"{file_size/1000000:.1f} MB"
            
            self.file_info_label.config(
                text=f"📄 {self.current_file.name} ({size_text})"
            )
    
    def upload_file(self):
        """Upload and analyze selected file."""
        if not self.current_file or not self.current_file.exists():
            messagebox.showwarning("No File", "Please select a file first.")
            return
        
        try:
            # Read file
            with open(self.current_file, 'r', encoding='utf-8') as f:
                self.original_text = f.read()
            
            # Update statistics
            self.stat_document_size.config(text=f"{len(self.original_text):,} chars")
            
            # Detect PII
            self.processing_queue.put(('processing_start', None))
            
            start_time = datetime.now()
            
            # Use model or regex for detection
            self.pii_entities = self.pii_processor.detect_pii(
                self.original_text, 
                self.use_model_detection
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate average confidence
            avg_confidence = np.mean([e['confidence'] for e in self.pii_entities]) if self.pii_entities else 0
            
            # Update statistics
            self.stat_pii_detected.config(text=str(len(self.pii_entities)))
            self.stat_avg_confidence.config(text=f"{avg_confidence:.1%}")
            self.stat_processing_time.config(text=f"{processing_time:.2f}s")
            
            # Display original text (NO COLORING)
            self.original_text_widget.delete('1.0', tk.END)
            self.original_text_widget.insert('1.0', self.original_text)
            self.original_text_widget.configure(font=('Segoe UI', self.text_font_size))
            
            # Clear redacted text
            self.redacted_text_widget.delete('1.0', tk.END)
            
            self.processing_queue.put(('processing_complete', None))
            
            # Show success message
            detection_mode = "ε=8.0 Model" if self.use_model_detection else "Regex Fallback"
            messagebox.showinfo("Success", 
                              f"File uploaded successfully.\n"
                              f"Detection Mode: {detection_mode}\n"
                              f"PII Detected: {len(self.pii_entities)}\n"
                              f"Average Confidence: {avg_confidence:.1%}\n"
                              f"Processing Time: {processing_time:.2f}s\n\n"
                              f"Select privacy level and click 'Apply Redaction'.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to upload file:\n{str(e)}")
            self.processing_queue.put(('processing_complete', None))
    
    def on_privacy_change(self, event=None):
        """Handle privacy level change."""
        try:
            self.current_epsilon = float(self.privacy_var.get())
            self.stat_privacy_level.config(text=f"ε={self.current_epsilon}")
            
            explanations = {
                0.5: "MAXIMUM PRIVACY",
                1.0: "HIGH PRIVACY", 
                2.0: "MODERATE-HIGH PRIVACY",
                3.0: "MODERATE PRIVACY",
                5.0: "MODERATE-LOW PRIVACY",
                8.0: "MAXIMUM ACCURACY"
            }
            
            if self.current_epsilon in explanations:
                self.status_label.config(text=f"Privacy: {explanations[self.current_epsilon]}")
                
        except:
            pass
    
    def redact_text(self):
        """Apply redaction based on current privacy level."""
        if not self.original_text:
            messagebox.showwarning("No Text", "Please upload a file first.")
            return
        
        if not self.pii_entities:
            messagebox.showinfo("No PII", "No PII detected in the document.")
            return
        
        try:
            self.processing_queue.put(('processing_start', None))
            
            start_time = datetime.now()
            
            # Apply redaction
            self.redacted_text = self.pii_processor.redact_text(
                self.original_text,
                self.pii_entities,
                self.current_epsilon
            )
            
            # Calculate redaction rate
            original_len = len(self.original_text)
            redacted_len = len(self.redacted_text.replace('█', ''))
            redaction_rate = ((original_len - redacted_len) / original_len * 100)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stat_redaction_rate.config(text=f"{redaction_rate:.1f}%")
            self.stat_processing_time.config(text=f"{processing_time:.2f}s")
            
            # Display redacted text (NO COLORING)
            self.redacted_text_widget.delete('1.0', tk.END)
            self.redacted_text_widget.insert('1.0', self.redacted_text)
            self.redacted_text_widget.configure(font=('Segoe UI', self.text_font_size))
            
            self.processing_queue.put(('processing_complete', None))
            
            # Show results
            detection_mode = "ε=8.0 Model" if self.use_model_detection else "Regex"
            messagebox.showinfo("Redaction Complete", 
                              f"Redaction applied successfully!\n\n"
                              f"Detection Mode: {detection_mode}\n"
                              f"Privacy Level: ε={self.current_epsilon}\n"
                              f"Redaction Rate: {redaction_rate:.1f}%\n"
                              f"Processing Time: {processing_time:.2f}s\n\n"
                              f"Click 'Save Redacted' to export the document.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Redaction failed:\n{str(e)}")
            self.processing_queue.put(('processing_complete', None))
    
    def save_redacted(self):
        """Save redacted text to file."""
        if not self.redacted_text:
            messagebox.showwarning("No Redacted Text", 
                                 "Please redact the text first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Redacted Document",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Add metadata header
                detection_mode = "ε=8.0 Model" if self.use_model_detection else "Regex Fallback"
                avg_confidence = np.mean([e['confidence'] for e in self.pii_entities]) if self.pii_entities else 0
                
                metadata = (
                    f"=== Redacted Document - Model-Based Detection ===\n"
                    f"Detection Mode: {detection_mode}\n"
                    f"Privacy Level: ε={self.current_epsilon}\n"
                    f"Original File: {self.current_file.name if self.current_file else 'Unknown'}\n"
                    f"Redaction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"PII Detected: {len(self.pii_entities)}\n"
                    f"Average Confidence: {avg_confidence:.1%}\n"
                    f"==================================================\n\n"
                )
                
                full_content = metadata + self.redacted_text
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(full_content)
                
                messagebox.showinfo("Success", 
                                  f"Redacted document saved to:\n{file_path}\n\n"
                                  f"Document includes detailed metadata header.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{str(e)}")
    
    def clear_all(self):
        """Clear all text and reset application."""
        self.original_text_widget.delete('1.0', tk.END)
        self.redacted_text_widget.delete('1.0', tk.END)
        
        self.current_file = None
        self.original_text = ""
        self.redacted_text = ""
        self.pii_entities = []
        
        self.file_info_label.config(text="No file selected")
        
        # Reset statistics
        self.stat_document_size.config(text="0 chars")
        self.stat_pii_detected.config(text="0")
        self.stat_avg_confidence.config(text="0%")
        self.stat_redaction_rate.config(text="0%")
        self.stat_processing_time.config(text="0.00s")
        self.stat_privacy_level.config(text=f"ε={self.current_epsilon}")
        
        self.set_status("green", "READY")
        messagebox.showinfo("Cleared", "All content has been cleared.\nReady for new document.")
    
    def run(self):
        """Start the application."""
        self.root.mainloop()

def main():
    """Main entry point."""
    root = tk.Tk()
    app = RedactorAppDemoLarge(root)
    app.run()

if __name__ == "__main__":
    main()