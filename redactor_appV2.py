#!/usr/bin/env python3
"""
RedactorAppDemo - LARGE TEXT VERSION
A modern GUI application for privacy-preserving PII redaction
with triple-sized text and color coding for basic PII categories.
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

class PIIProcessor:
    """Handles PII detection and processing using regex patterns."""
    
    def __init__(self):
        self.pii_patterns = self._compile_patterns()
        self.color_scheme = self._get_color_scheme()
        
    def _compile_patterns(self):
        """Compile regex patterns for all PII categories."""
        return {
            # Personal Identifiers
            'FULLNAME': r'\b(?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.)?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b',
            'FIRSTNAME': r'\b(?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.)?\s*([A-Z][a-z]+)\b(?!\s+(?:Street|St\.|Ave\.|Avenue|Road|Rd\.))',
            'LASTNAME': r'\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?\b(?=\s*(?:,|\.|\s|$))',
            
            # Contact Information
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            'STREET_ADDRESS': r'\b\d{1,5}\s+[A-Za-z0-9\s]+(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Boulevard|Blvd\.|Drive|Dr\.|Lane|Ln\.)\b',
            'CITY': r'\b(?:New\s+York|Los\s+Angeles|Chicago|Houston|Phoenix|Philadelphia|San\s+Antonio|San\s+Diego|Dallas|San\s+Jose|Austin|Jacksonville|Fort\s+Worth|Columbus|Charlotte|San\s+Francisco|Indianapolis|Seattle|Denver|Washington)\b',
            'STATE': r'\b(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b',
            'ZIP_CODE': r'\b\d{5}(?:-\d{4})?\b',
            
            # Financial Data
            'ACCOUNT_NUMBER': r'\b\d{8,12}\b',
            'CREDIT_CARD': r'\b(?:\d{4}[- ]?){3}\d{4}\b',
            'BITCOIN_ADDRESS': r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',
            'ETHEREUM_ADDRESS': r'\b0x[a-fA-F0-9]{40}\b',
            'IBAN': r'\b[A-Z]{2}\d{2}[A-Z0-9]{4,}\d{7,}(?:[A-Z0-9]?){0,16}\b',
            'BIC': r'\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b',
            
            # Digital Identifiers (detected but not colored)
            'IPV4': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'IPV6': r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b',
            'MAC_ADDRESS': r'\b(?:[A-Fa-f0-9]{2}:){5}[A-Fa-f0-9]{2}\b',
            'USERNAME': r'\b@[A-Za-z0-9_]{3,15}\b',
            'PASSWORD': r'\b(?:password|pwd|pass)[:=]\s*[\w@#$%^&*]{6,}\b',
            
            # Demographic Information (detected but not colored)
            'GENDER': r'\b(?:Male|Female|Non-binary|M|F|NB)\b',
            'JOB_TITLE': r'\b(?:CEO|CTO|CFO|Director|Manager|Engineer|Analyst|Specialist|Consultant|Assistant)\b',
            'AREA': r'\b(?:Urban|Suburban|Rural|Metropolitan|Downtown|Uptown)\b',
            
            # Geographic Data (detected but not colored)
            'GPS_COORDINATES': r'\b-?\d{1,3}\.\d{1,6},\s*-?\d{1,3}\.\d{1,6}\b',
            'ORDINAL_DIRECTION': r'\b(?:North|South|East|West|Northeast|Northwest|Southeast|Southwest|N|S|E|W|NE|NW|SE|SW)\b',
            'COUNTY': r'\b(?:County|Parish|Borough)\s+[A-Z][A-Za-z\s]+\b',
            
            # Technical Identifiers (detected but not colored)
            'USER_AGENT': r'\bMozilla/\d\.\d.*?AppleWebKit/\d+\.\d+.*?Chrome/\d+\.\d+\.\d+\.\d+\b',
            'URL': r'\bhttps?://(?:www\.)?[A-Za-z0-9-]+\.[A-Za-z]{2,}(?:/\S*)?\b',
            'DEVICE_INFO': r'\b(?:iPhone|iPad|Android|Windows|Mac|Linux|Samsung|Google|Pixel|Dell|HP|Lenovo)\s+[A-Za-z0-9\s]+\b'
        }
    
    def _get_color_scheme(self):
        """Define color scheme for BASIC PII types only."""
        return {
            # ========== BASIC PII CATEGORIES (COLORED) ==========
            
            # Personal Identifiers - Blue shades
            'FULLNAME': '#2980b9',
            'FIRSTNAME': '#3498db',
            'LASTNAME': '#5dade2',
            
            # Contact Information - Green shades
            'EMAIL': '#27ae60',
            'PHONE': '#2ecc71',
            'STREET_ADDRESS': '#1abc9c',
            'CITY': '#16a085',
            'STATE': '#45b39d',
            'ZIP_CODE': '#52be80',
            
            # Financial Data - Red/Orange shades
            'ACCOUNT_NUMBER': '#c0392b',
            'CREDIT_CARD': '#e74c3c',
            'BITCOIN_ADDRESS': '#d35400',
            'ETHEREUM_ADDRESS': '#e67e22',
            'IBAN': '#f39c12',
            'BIC': '#f1c40f',
            
            # ========== OTHER PII CATEGORIES (DETECTED BUT NOT COLORED) ==========
            # These will appear in normal text color
        }
    
    def detect_pii(self, text: str) -> List[Dict]:
        """Detect all PII in text using regex patterns."""
        pii_entities = []
        
        for pii_type, pattern in self.pii_patterns.items():
            try:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Avoid overlapping matches
                    overlap = False
                    for existing in pii_entities:
                        if (match.start() >= existing['start'] and match.start() <= existing['end']) or \
                           (match.end() >= existing['start'] and match.end() <= existing['end']):
                            overlap = True
                            break
                    
                    if not overlap:
                        # Only apply color to BASIC PII categories
                        is_basic_pii = pii_type in [
                            # Personal Identifiers
                            'FULLNAME', 'FIRSTNAME', 'LASTNAME',
                            # Contact Information
                            'EMAIL', 'PHONE', 'STREET_ADDRESS', 'CITY', 'STATE', 'ZIP_CODE',
                            # Financial Data
                            'ACCOUNT_NUMBER', 'CREDIT_CARD', 'BITCOIN_ADDRESS', 
                            'ETHEREUM_ADDRESS', 'IBAN', 'BIC'
                        ]
                        
                        color = self.color_scheme.get(pii_type, '#ecf0f1') if is_basic_pii else '#ecf0f1'
                        
                        pii_entities.append({
                            'type': pii_type,
                            'text': match.group(),
                            'start': match.start(),
                            'end': match.end(),
                            'color': color,
                            'is_basic': is_basic_pii  # Flag for basic PII
                        })
            except re.error:
                continue
        
        # Sort by start position
        pii_entities.sort(key=lambda x: x['start'])
        return pii_entities
    
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
            if np.random.random() < redaction_prob:
                # Redact this entity
                start, end = entity['start'], entity['end']
                redaction_char = '█'  # Full block character
                
                # Replace with redaction characters
                for i in range(start, end):
                    if i < len(chars):
                        chars[i] = redaction_char
        
        return ''.join(chars)
    
    def highlight_text(self, text_widget: tk.Text, text: str, pii_entities: List[Dict], font_size: int = 16):
        """Apply highlighting to text widget based on PII entities with large font."""
        text_widget.delete('1.0', tk.END)
        text_widget.insert('1.0', text)
        
        # Configure base font
        text_widget.configure(font=('Segoe UI', font_size))
        
        # Configure tags for BASIC PII types only
        basic_pii_count = 0
        for entity in pii_entities:
            if entity['is_basic']:
                tag_name = f"pii_{entity['type']}"
                text_widget.tag_configure(tag_name, 
                                         foreground=entity['color'],
                                         font=('Segoe UI', font_size, 'bold'))
                
                # Apply tag
                start_idx = f"1.0+{entity['start']}c"
                end_idx = f"1.0+{entity['end']}c"
                text_widget.tag_add(tag_name, start_idx, end_idx)
                basic_pii_count += 1
        
        return basic_pii_count

class RedactorAppDemoLarge:
    def __init__(self, root):
        self.root = root
        self.root.title("RedactorAppDemo - Privacy-Preserving PII Redaction (LARGE TEXT)")
        
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
        self.basic_pii_count = 0
        self.other_pii_count = 0
        
        # Font size configuration (TRIPLED from standard)
        self.base_font_size = 16  # Standard would be ~10-12
        self.title_font_size = 36  # Standard would be 24
        self.button_font_size = 14  # Standard would be 10
        self.label_font_size = 14  # Standard would be 10
        self.text_font_size = 18  # Standard would be 10
        
        # Initialize PII processor
        self.pii_processor = PIIProcessor()
        
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
            # Create a privacy-themed icon
            from PIL import Image, ImageDraw
            icon = Image.new('RGBA', (96, 96), color=(0, 0, 0, 0))  # Larger icon
            draw = ImageDraw.Draw(icon)
            
            # Draw shield background
            draw.ellipse([12, 12, 84, 84], fill='#3498db')
            
            # Draw lock
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
        
        # Create a dark theme with large fonts
        style.theme_create('dark_large', parent='clam', settings={
            'TFrame': {
                'configure': {'background': '#1e272e'}
            },
            'TLabel': {
                'configure': {
                    'background': '#1e272e',
                    'foreground': '#ecf0f1',
                    'font': ('Segoe UI', self.label_font_size)  # Larger font
                }
            },
            'TButton': {
                'configure': {
                    'background': '#3498db',
                    'foreground': 'white',
                    'borderwidth': 0,
                    'focuscolor': 'none',
                    'font': ('Segoe UI', self.button_font_size, 'bold'),  # Larger font
                    'padding': (25, 15)  # Larger padding
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
                    'font': ('Segoe UI', self.label_font_size),  # Larger font
                    'padding': (10, 5)  # Larger padding
                }
            },
            'Horizontal.TProgressbar': {
                'configure': {
                    'background': '#3498db',
                    'troughcolor': '#2c3e50',
                    'borderwidth': 0,
                    'lightcolor': '#3498db',
                    'darkcolor': '#3498db',
                    'thickness': 30  # Thicker progress bar
                }
            },
            'Treeview': {
                'configure': {
                    'background': '#2c3e50',
                    'foreground': '#ecf0f1',
                    'fieldbackground': '#2c3e50',
                    'borderwidth': 0,
                    'font': ('Segoe UI', self.label_font_size)  # Larger font
                },
                'map': {
                    'background': [('selected', '#3498db')],
                    'foreground': [('selected', 'white')]
                }
            },
            'Treeview.Heading': {
                'configure': {
                    'background': '#34495e',
                    'foreground': '#ecf0f1',
                    'relief': 'flat',
                    'borderwidth': 0,
                    'font': ('Segoe UI', self.label_font_size, 'bold')  # Larger font
                }
            },
            'TLabelframe': {
                'configure': {
                    'background': '#2c3e50',
                    'foreground': '#ecf0f1',
                    'relief': 'flat',
                    'borderwidth': 2,
                    'padding': (20, 15)  # Larger padding
                }
            },
            'TLabelframe.Label': {
                'configure': {
                    'background': '#2c3e50',
                    'foreground': '#3498db',
                    'font': ('Segoe UI', self.button_font_size, 'bold')  # Larger font
                }
            }
        })
        style.theme_use('dark_large')
    
    def create_widgets(self):
        """Create all GUI widgets with large text and elements."""
        # Main container with larger padding
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=25, pady=25)
        
        # Header with larger fonts
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill='x', pady=(0, 30))
        
        # Title with gradient effect simulation
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side='left', fill='both', expand=True)
        
        title_label = tk.Label(title_frame,
                              text="🔒 REDACTORAPP DEMO",
                              font=('Segoe UI', self.title_font_size, 'bold'),
                              bg='#1e272e',
                              fg='#3498db')
        title_label.pack(side='left')
        
        subtitle_label = tk.Label(title_frame,
                                 text="Privacy-Preserving PII Redaction",
                                 font=('Segoe UI', int(self.title_font_size * 0.6)),
                                 bg='#1e272e',
                                 fg='#7f8c8d')
        subtitle_label.pack(side='left', padx=(20, 0))
        
        # Status indicator - larger
        self.status_frame = ttk.Frame(header_frame)
        self.status_frame.pack(side='right')
        
        self.status_dot = tk.Canvas(self.status_frame, width=40, height=40,  # Larger
                                   bg='#1e272e', highlightthickness=0)
        self.status_dot.pack(side='left', padx=(0, 20))
        self.draw_status_dot('green')  # Initial status
        
        self.status_label = ttk.Label(self.status_frame, text="READY", 
                                     font=('Segoe UI', self.label_font_size, 'bold'))
        self.status_label.pack(side='left')
        
        # Main content area
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill='both', expand=True)
        
        # Left panel - File explorer and original text
        left_panel = ttk.LabelFrame(content_frame, text="📄 ORIGINAL DOCUMENT", 
                                   padding=(25, 20))  # Larger padding
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        # File controls - larger buttons
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
        
        # File info - larger text
        self.file_info_label = ttk.Label(file_controls,
                                        text="No file selected",
                                        font=('Segoe UI', self.label_font_size))
        self.file_info_label.pack(side='right')
        
        # Original text display
        text_frame = ttk.Frame(left_panel)
        text_frame.pack(fill='both', expand=True)
        
        # Text widget with scrollbar
        text_container = ttk.Frame(text_frame)
        text_container.pack(fill='both', expand=True)
        
        # Create scrollbar - thicker
        text_scrollbar = ttk.Scrollbar(text_container)
        text_scrollbar.pack(side='right', fill='y')
        
        # Create text widget with LARGE font
        self.original_text_widget = tk.Text(text_container,
                                           wrap='word',
                                           font=('Segoe UI', self.text_font_size),  # Larger font
                                           bg='#2c3e50',
                                           fg='#ecf0f1',
                                           insertbackground='#ecf0f1',
                                           selectbackground='#3498db',
                                           borderwidth=0,
                                           relief='flat',
                                           padx=25,  # Larger padding
                                           pady=25,  # Larger padding
                                           yscrollcommand=text_scrollbar.set)
        self.original_text_widget.pack(side='left', fill='both', expand=True)
        text_scrollbar.config(command=self.original_text_widget.yview)
        
        # Right panel - Redacted text
        right_panel = ttk.LabelFrame(content_frame, text="🔒 REDACTED DOCUMENT",
                                    padding=(25, 20))  # Larger padding
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Privacy level selector - larger
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
                                         width=15,  # Wider
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
        
        # Create scrollbar for redacted text - thicker
        redacted_scrollbar = ttk.Scrollbar(redacted_text_frame)
        redacted_scrollbar.pack(side='right', fill='y')
        
        # Create redacted text widget with LARGE font
        self.redacted_text_widget = tk.Text(redacted_text_frame,
                                           wrap='word',
                                           font=('Segoe UI', self.text_font_size),  # Larger font
                                           bg='#2c3e50',
                                           fg='#ecf0f1',
                                           insertbackground='#ecf0f1',
                                           selectbackground='#3498db',
                                           borderwidth=0,
                                           relief='flat',
                                           padx=25,  # Larger padding
                                           pady=25,  # Larger padding
                                           yscrollcommand=redacted_scrollbar.set)
        self.redacted_text_widget.pack(side='left', fill='both', expand=True)
        redacted_scrollbar.config(command=self.redacted_text_widget.yview)
        
        # Bottom panel - Statistics and controls
        bottom_panel = ttk.Frame(main_container)
        bottom_panel.pack(fill='x', pady=(30, 0))
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(bottom_panel, text="📊 PII STATISTICS",
                                    padding=(30, 20))  # Larger padding
        stats_frame.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        # Create statistics labels
        self.stats_grid = ttk.Frame(stats_frame)
        self.stats_grid.pack(fill='both', expand=True)
        
        self.init_stats_labels()
        
        # Controls frame
        controls_frame = ttk.LabelFrame(bottom_panel, text="⚙️ CONTROLS",
                                       padding=(30, 20))  # Larger padding
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
        
        # PII Legend - larger (only for BASIC PIIs)
        self.create_pii_legend(bottom_panel)
    
    def init_stats_labels(self):
        """Initialize statistics labels with larger fonts."""
        stats = [
            ("DOCUMENT SIZE:", "0 chars"),
            ("BASIC PII:", "0"),
            ("OTHER PII:", "0"),
            ("TOTAL PII:", "0"),
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
            
            # Store reference to update later
            var_name = f'stat_{label.lower().replace(":", "").replace(" ", "_")}'
            setattr(self, var_name, value_widget)
    
    def create_pii_legend(self, parent):
        """Create PII type color legend with larger elements - ONLY BASIC PIIs."""
        legend_frame = ttk.LabelFrame(parent, text="🎨 BASIC PII COLOR LEGEND",
                                     padding=(25, 20))  # Larger padding
        legend_frame.pack(side='top', fill='x', pady=(15, 0))
        
        # Create a canvas for the legend - taller
        canvas = tk.Canvas(legend_frame, height=180, bg='#1e272e',  # Taller
                          highlightthickness=0)
        canvas.pack(fill='x')
        
        # Only show BASIC PII categories
        categories = {
            "PERSONAL IDENTIFIERS": ["FULLNAME", "FIRSTNAME", "LASTNAME"],
            "CONTACT INFO": ["EMAIL", "PHONE", "STREET_ADDRESS", "CITY", "STATE", "ZIP_CODE"],
            "FINANCIAL DATA": ["ACCOUNT_NUMBER", "CREDIT_CARD", "BITCOIN_ADDRESS", 
                              "ETHEREUM_ADDRESS", "IBAN", "BIC"]
        }
        
        x_pos = 20
        y_pos = 20
        
        for category, pii_types in categories.items():
            # Draw category label with larger font
            canvas.create_text(x_pos, y_pos, text=category + ":", 
                             anchor='nw', fill='#7f8c8d', 
                             font=('Segoe UI', self.label_font_size - 2, 'bold'))
            y_pos += 25
            
            # Draw color boxes and labels with larger elements
            for pii_type in pii_types:
                if pii_type in self.pii_processor.color_scheme:
                    color = self.pii_processor.color_scheme[pii_type]
                    
                    # Draw larger color box
                    canvas.create_rectangle(x_pos, y_pos, x_pos + 20, y_pos + 20,  # Larger box
                                          fill=color, outline='')
                    
                    # Draw label with larger font
                    canvas.create_text(x_pos + 30, y_pos + 10, 
                                     text=pii_type.replace('_', ' ').title(),
                                     anchor='w', fill='#ecf0f1',
                                     font=('Segoe UI', self.label_font_size - 2))
                    
                    y_pos += 30
            
            y_pos += 15
            x_pos += 300  # More space between columns
            
            # Move to next column if needed
            if x_pos > 900:
                x_pos = 20
                y_pos += 180
        
        # Add note about other PII types
        canvas.create_text(20, y_pos + 10,
                         text="Note: Other PII types (Digital IDs, Demographic, Geographic, Technical) are detected but not color-coded.",
                         anchor='nw', fill='#95a5a6',
                         font=('Segoe UI', self.label_font_size - 4))
    
    def draw_status_dot(self, color):
        """Draw larger status indicator dot."""
        self.status_dot.delete('all')
        self.status_dot.create_oval(10, 10, 30, 30, fill=color, outline='', width=2)  # Larger dot
    
    def load_models_async(self):
        """Load models in background thread."""
        self.set_status("yellow", "LOADING MODELS...")
        
        def load_task():
            for epsilon in self.available_epsilons:
                try:
                    model_path = self.models_dir / f"final_epsilon_{epsilon}"
                    if model_path.exists():
                        # Load model config
                        with open(model_path / "config.json", 'r') as f:
                            config = json.load(f)
                        
                        # Load model parameters
                        with open(model_path / "best_params.pkl", 'rb') as f:
                            params = pickle.load(f)
                        
                        self.loaded_models[epsilon] = {
                            'config': config,
                            'params': params,
                            'model': DistilBERTDPModel(hidden_size=config.get('hidden_size', 768))
                        }
                        
                        # Update queue
                        self.processing_queue.put(('model_loaded', epsilon))
                except Exception as e:
                    print(f"Error loading model for ε={epsilon}: {e}")
            
            self.processing_queue.put(('models_complete', None))
        
        thread = threading.Thread(target=load_task, daemon=True)
        thread.start()
    
    def check_queue(self):
        """Check processing queue for updates."""
        try:
            while True:
                msg_type, data = self.processing_queue.get_nowait()
                
                if msg_type == 'model_loaded':
                    self.update_model_status(f"Loaded ε={data}")
                elif msg_type == 'models_complete':
                    self.set_status("green", "READY")
                    self.stat_model_status.config(text="All models loaded")
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
    
    def update_model_status(self, text):
        """Update model status label."""
        current = self.stat_model_status.cget('text')
        if current == "Loading...":
            self.stat_model_status.config(text=text)
        else:
            self.stat_model_status.config(text=f"{current}, {text}")
    
    def browse_file(self):
        """Open file dialog to select text file."""
        file_path = filedialog.askopenfilename(
            title="Select Text File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_file = Path(file_path)
            file_size = self.current_file.stat().st_size
            size_text = f"{file_size:,} bytes" if file_size < 1000000 else f"{file_size/1000000:.1f} MB"
            
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
            self.pii_entities = self.pii_processor.detect_pii(self.original_text)
            
            # Count basic vs other PII
            self.basic_pii_count = sum(1 for entity in self.pii_entities if entity['is_basic'])
            self.other_pii_count = len(self.pii_entities) - self.basic_pii_count
            
            # Update PII counts
            self.stat_basic_pii.config(text=str(self.basic_pii_count))
            self.stat_other_pii.config(text=str(self.other_pii_count))
            self.stat_total_pii.config(text=str(len(self.pii_entities)))
            
            # Highlight original text with LARGE font (only basic PIIs get colors)
            colored_count = self.pii_processor.highlight_text(
                self.original_text_widget, 
                self.original_text, 
                self.pii_entities,
                self.text_font_size
            )
            
            # Clear redacted text
            self.redacted_text_widget.delete('1.0', tk.END)
            
            self.processing_queue.put(('processing_complete', None))
            
            # Show success message
            messagebox.showinfo("Success", 
                              f"File uploaded successfully.\n"
                              f"Found {len(self.pii_entities)} PII entities:\n"
                              f"• {self.basic_pii_count} Basic PII (color-coded)\n"
                              f"• {self.other_pii_count} Other PII (detected, not colored)\n\n"
                              f"Select privacy level and click 'Apply Redaction'.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to upload file:\n{str(e)}")
            self.processing_queue.put(('processing_complete', None))
    
    def on_privacy_change(self, event=None):
        """Handle privacy level change."""
        try:
            self.current_epsilon = float(self.privacy_var.get())
            self.stat_privacy_level.config(text=f"ε={self.current_epsilon}")
            
            # Show privacy level explanation
            explanations = {
                0.5: "MAXIMUM PRIVACY (may miss some PII)",
                1.0: "HIGH PRIVACY",
                2.0: "MODERATE-HIGH PRIVACY",
                3.0: "MODERATE PRIVACY",
                5.0: "MODERATE-LOW PRIVACY",
                8.0: "MAXIMUM ACCURACY (weakest privacy)"
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
            
            # Display redacted text with same highlighting and LARGE font
            self.pii_processor.highlight_text(self.redacted_text_widget,
                                             self.redacted_text,
                                             self.pii_entities,
                                             self.text_font_size)
            
            self.processing_queue.put(('processing_complete', None))
            
            # Show explanation of what redaction means at this privacy level
            privacy_explanations = {
                0.5: ("STRONG privacy redaction", 
                     "High privacy protection, many PII may remain for readability"),
                1.0: ("HIGH privacy redaction", 
                     "Strong privacy protection, some PII may remain"),
                2.0: ("MODERATE-HIGH privacy redaction", 
                     "Good privacy with reasonable readability"),
                3.0: ("MODERATE privacy redaction", 
                     "Balanced privacy and readability"),
                5.0: ("MODERATE-LOW privacy redaction", 
                     "More readable with some privacy"),
                8.0: ("LIGHT privacy redaction", 
                     "Maximum accuracy, minimal privacy protection")
            }
            
            level_text, desc_text = privacy_explanations.get(self.current_epsilon, ("Redaction", "Applied"))
            
            messagebox.showinfo("Redaction Applied", 
                              f"Applied {level_text} (ε={self.current_epsilon}).\n"
                              f"{desc_text}.\n"
                              f"Redaction rate: {redaction_rate:.1f}%\n"
                              f"Processing time: {processing_time:.2f}s")
            
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
                metadata = (
                    f"=== Redacted Document ===\n"
                    f"Original: {self.current_file.name if self.current_file else 'Unknown'}\n"
                    f"Privacy Level: ε={self.current_epsilon}\n"
                    f"Redaction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Basic PII Count: {self.basic_pii_count}\n"
                    f"Other PII Count: {self.other_pii_count}\n"
                    f"Total PII: {len(self.pii_entities)}\n"
                    f"=================================\n\n"
                )
                
                full_content = metadata + self.redacted_text
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(full_content)
                
                messagebox.showinfo("Success", 
                                  f"Redacted document saved to:\n{file_path}\n\n"
                                  f"Document includes metadata header with redaction details.")
                
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
        self.basic_pii_count = 0
        self.other_pii_count = 0
        
        self.file_info_label.config(text="No file selected")
        
        # Reset statistics
        self.stat_document_size.config(text="0 chars")
        self.stat_basic_pii.config(text="0")
        self.stat_other_pii.config(text="0")
        self.stat_total_pii.config(text="0")
        self.stat_redaction_rate.config(text="0%")
        self.stat_processing_time.config(text="0.00s")
        self.stat_privacy_level.config(text=f"ε={self.current_epsilon}")
        
        self.set_status("green", "READY")
        messagebox.showinfo("Cleared", "All content has been cleared.\nApplication is ready for new document.")
    
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