# Create the redaction pipeline
mkdir -p src/redaction

cat > src/redaction/redaction_pipeline.py << 'EOF'
#!/usr/bin/env python3
"""
Functional Redaction Pipeline for PII detection and masking.
Uses the best DP model (ε=8.0) to detect PII in text.
"""
import os
import sys
import pickle
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from jax import tree_util

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ========== MODEL DEFINITION (must match training) ==========
class RedactionModel(nn.Module):
    """Model architecture for PII detection (must match training)."""
    hidden_size: int
    num_classes: int = 2
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = jnp.tanh(x)
        x = nn.Dense(self.hidden_size // 2)(x)
        x = jnp.tanh(x)
        x = nn.Dense(self.num_classes)(x)
        return x

# ========== PII CATEGORIES ==========
class PIICategory(Enum):
    """Categories of PII that can be detected."""
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    NAME = "NAME"
    ADDRESS = "ADDRESS"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    DATE = "DATE"
    IP_ADDRESS = "IP_ADDRESS"
    GENERIC = "GENERIC_PII"

@dataclass
class PIIEntity:
    """Represents a detected PII entity in text."""
    text: str
    category: PIICategory
    start_pos: int
    end_pos: int
    confidence: float
    
    def to_dict(self):
        return {
            "text": self.text,
            "category": self.category.value,
            "start": self.start_pos,
            "end": self.end_pos,
            "confidence": float(self.confidence)
        }

# ========== FEATURE EXTRACTION ==========
class FeatureExtractor:
    """Extracts features from text for PII detection."""
    
    @staticmethod
    def extract_features_from_text(text: str) -> np.ndarray:
        """Extract features from text (must match training features)."""
        # Same features as used in training
        length = min(len(text) / 50, 2.0)
        
        # Email patterns
        has_at = 1.0 if '@' in text else 0.0
        has_dot_after_at = 1.0 if '@' in text and '.' in text.split('@')[-1] else 0.0
        
        # Number patterns
        digits = sum(c.isdigit() for c in text)
        digit_ratio = digits / max(len(text), 1)
        
        # Capitalization patterns
        words = text.split()
        has_caps = 1.0 if any(w.istitle() for w in words) else 0.0
        
        features = np.array([
            length,
            has_at,
            has_dot_after_at,
            digit_ratio,
            has_caps,
            len(words) / 20.0
        ], dtype=np.float32)
        
        return features.reshape(1, -1)
    
    @staticmethod
    def extract_features_from_chunk(chunk: str, context: str = "") -> np.ndarray:
        """Extract features from a text chunk with context."""
        # Combine chunk with context for better detection
        full_text = f"{context} {chunk}" if context else chunk
        return FeatureExtractor.extract_features_from_text(full_text)

# ========== MODEL LOADER ==========
class ModelLoader:
    """Loads and manages the PII detection model."""
    
    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            # Default to best model (ε=8.0)
            model_path = project_root / "outputs/models/final_epsilon_8.0"
        
        self.model_path = Path(model_path)
        self.model = None
        self.params = None
        self.config = None
        self.scaler = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and configuration."""
        print(f"🔍 Loading model from: {self.model_path}")
        
        # Load parameters
        params_path = self.model_path / "best_params.pkl"
        if not params_path.exists():
            params_path = self.model_path / "params.pkl"
        
        if not params_path.exists():
            raise FileNotFoundError(f"Model parameters not found in {self.model_path}")
        
        with open(params_path, 'rb') as f:
            self.params = pickle.load(f)
        
        # Load configuration
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.hidden_size = self.config.get('hidden_size', 32)
            self.input_dim = self.config.get('input_dim', 6)
        else:
            # Default values
            self.hidden_size = 32
            self.input_dim = 6
        
        # Create model
        self.model = RedactionModel(hidden_size=self.hidden_size)
        
        print(f"✅ Model loaded: {self.hidden_size} hidden units, {self.input_dim} features")
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on features."""
        # Ensure features have correct dimension
        if features.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {features.shape[1]}")
        
        # Convert to JAX array
        features_jax = jnp.array(features, dtype=jnp.float32)
        
        # Make prediction
        logits = self.model.apply({'params': self.params}, features_jax)
        probabilities = jax.nn.softmax(logits, axis=-1)
        predictions = jnp.argmax(logits, axis=-1)
        
        return np.array(predictions), np.array(probabilities)

# ========== PII DETECTORS ==========
class PIIDetector:
    """Main PII detector using ML model and rule-based heuristics."""
    
    def __init__(self, model_loader: Optional[ModelLoader] = None):
        self.model_loader = model_loader or ModelLoader()
        self.regex_patterns = self._compile_regex_patterns()
    
    def _compile_regex_patterns(self) -> Dict[PIICategory, re.Pattern]:
        """Compile regex patterns for different PII categories."""
        return {
            PIICategory.EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                re.IGNORECASE
            ),
            PIICategory.PHONE: re.compile(
                r'\b(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'
            ),
            PIICategory.SSN: re.compile(
                r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'
            ),
            PIICategory.CREDIT_CARD: re.compile(
                r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
            ),
            PIICategory.IP_ADDRESS: re.compile(
                r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
            ),
            PIICategory.DATE: re.compile(
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'
            )
        }
    
    def detect_with_model(self, text: str, chunk_size: int = 50) -> List[PIIEntity]:
        """Detect PII using the ML model."""
        entities = []
        
        # Split text into overlapping chunks
        chunks = self._create_text_chunks(text, chunk_size)
        
        for i, chunk in enumerate(chunks):
            # Get context (previous and next chunks)
            context = ""
            if i > 0:
                context = chunks[i-1]
            if i < len(chunks) - 1:
                context += " " + chunks[i+1]
            
            # Extract features
            features = FeatureExtractor.extract_features_from_chunk(chunk, context)
            
            # Make prediction
            predictions, probabilities = self.model_loader.predict(features)
            
            if predictions[0] == 1:  # PII detected
                confidence = probabilities[0][1]
                
                # Determine category using regex
                category = self._classify_pii_category(chunk)
                
                # Find position in original text
                start_pos = text.find(chunk)
                if start_pos != -1:
                    entity = PIIEntity(
                        text=chunk,
                        category=category,
                        start_pos=start_pos,
                        end_pos=start_pos + len(chunk),
                        confidence=confidence
                    )
                    entities.append(entity)
        
        return self._merge_overlapping_entities(entities)
    
    def detect_with_regex(self, text: str) -> List[PIIEntity]:
        """Detect PII using regex patterns."""
        entities = []
        
        for category, pattern in self.regex_patterns.items():
            for match in pattern.finditer(text):
                entity = PIIEntity(
                    text=match.group(),
                    category=category,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=1.0  # High confidence for regex matches
                )
                entities.append(entity)
        
        return self._merge_overlapping_entities(entities)
    
    def detect_hybrid(self, text: str) -> List[PIIEntity]:
        """Detect PII using both ML model and regex (ensemble)."""
        ml_entities = self.detect_with_model(text)
        regex_entities = self.detect_with_regex(text)
        
        # Combine and deduplicate
        all_entities = ml_entities + regex_entities
        return self._deduplicate_entities(all_entities)
    
    def _create_text_chunks(self, text: str, chunk_size: int = 50) -> List[str]:
        """Split text into overlapping chunks for analysis."""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size // 5):  # 80% overlap
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunks.append(' '.join(chunk_words))
        
        return chunks
    
    def _classify_pii_category(self, text: str) -> PIICategory:
        """Classify PII into specific categories."""
        text_lower = text.lower()
        
        # Check each regex pattern
        for category, pattern in self.regex_patterns.items():
            if pattern.search(text):
                return category
        
        # Heuristic classification
        if '@' in text and '.' in text.split('@')[-1]:
            return PIICategory.EMAIL
        elif any(c.isdigit() for c in text) and sum(c.isdigit() for c in text) > 7:
            return PIICategory.PHONE
        elif any(word.istitle() for word in text.split()):
            return PIICategory.NAME
        elif any(word.lower() in ['street', 'st', 'avenue', 'ave', 'road', 'rd'] 
                for word in text.split()):
            return PIICategory.ADDRESS
        else:
            return PIICategory.GENERIC
    
    def _merge_overlapping_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Merge overlapping PII entities."""
        if not entities:
            return []
        
        # Sort by start position
        entities.sort(key=lambda x: x.start_pos)
        
        merged = []
        current = entities[0]
        
        for entity in entities[1:]:
            if entity.start_pos <= current.end_pos:
                # Overlapping, merge
                current.end_pos = max(current.end_pos, entity.end_pos)
                current.text = current.text + text[current.end_pos:entity.end_pos]
                # Keep higher confidence category
                if entity.confidence > current.confidence:
                    current.category = entity.category
                    current.confidence = entity.confidence
            else:
                merged.append(current)
                current = entity
        
        merged.append(current)
        return merged
    
    def _deduplicate_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Remove duplicate entities."""
        seen_positions = set()
        unique_entities = []
        
        for entity in entities:
            position_key = (entity.start_pos, entity.end_pos)
            if position_key not in seen_positions:
                seen_positions.add(position_key)
                unique_entities.append(entity)
        
        return unique_entities

# ========== REDACTION ENGINE ==========
class RedactionEngine:
    """Engine for redacting PII from text."""
    
    def __init__(self, detector: Optional[PIIDetector] = None):
        self.detector = detector or PIIDetector()
        self.redaction_styles = {
            'mask': '[REDACTED]',
            'hash': '####',
            'partial': self._partial_redaction,
            'category_specific': self._category_specific_redaction
        }
    
    def redact_text(self, text: str, style: str = 'mask', 
                   categories: Optional[List[PIICategory]] = None) -> Tuple[str, List[PIIEntity]]:
        """
        Redact PII from text.
        
        Args:
            text: Input text containing PII
            style: Redaction style ('mask', 'hash', 'partial', 'category_specific')
            categories: Specific PII categories to redact (None = all)
        
        Returns:
            Tuple of (redacted_text, list_of_detected_entities)
        """
        # Detect PII
        entities = self.detector.detect_hybrid(text)
        
        # Filter by categories if specified
        if categories:
            entities = [e for e in entities if e.category in categories]
        
        # Sort entities by start position (descending for safe replacement)
        entities.sort(key=lambda x: x.start_pos, reverse=True)
        
        # Apply redaction
        redacted_text = text
        for entity in entities:
            redaction = self._get_redaction_text(entity, style)
            redacted_text = (
                redacted_text[:entity.start_pos] + 
                redaction + 
                redacted_text[entity.end_pos:]
            )
        
        return redacted_text, entities
    
    def _get_redaction_text(self, entity: PIIEntity, style: str) -> str:
        """Get the redaction text for an entity based on style."""
        if style in self.redaction_styles:
            if callable(self.redaction_styles[style]):
                return self.redaction_styles[style](entity)
            else:
                return self.redaction_styles[style]
        else:
            return '[REDACTED]'
    
    def _partial_redaction(self, entity: PIIEntity) -> str:
        """Partially redact the entity (e.g., j***@example.com)."""
        text = entity.text
        
        if entity.category == PIICategory.EMAIL:
            if '@' in text:
                local, domain = text.split('@', 1)
                if len(local) > 2:
                    return f"{local[0]}***@{domain}"
        
        elif entity.category == PIICategory.PHONE:
            # Keep last 4 digits
            digits = ''.join(filter(str.isdigit, text))
            if len(digits) >= 4:
                return f"***-***-{digits[-4:]}"
        
        # Default partial redaction
        if len(text) > 4:
            return f"{text[0]}***{text[-1]}"
        else:
            return "***"
    
    def _category_specific_redaction(self, entity: PIIEntity) -> str:
        """Category-specific redaction labels."""
        labels = {
            PIICategory.EMAIL: '[EMAIL]',
            PIICategory.PHONE: '[PHONE]',
            PIICategory.NAME: '[NAME]',
            PIICategory.ADDRESS: '[ADDRESS]',
            PIICategory.SSN: '[SSN]',
            PIICategory.CREDIT_CARD: '[CREDIT_CARD]',
            PIICategory.DATE: '[DATE]',
            PIICategory.IP_ADDRESS: '[IP_ADDRESS]',
            PIICategory.GENERIC: '[PII]'
        }
        return labels.get(entity.category, '[PII]')

# ========== PIPELINE ORCHESTRATOR ==========
class RedactionPipeline:
    """Orchestrates the complete redaction pipeline."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_loader = ModelLoader(model_path)
        self.detector = PIIDetector(self.model_loader)
        self.engine = RedactionEngine(self.detector)
        
        print("🚀 Redaction Pipeline Initialized")
        print(f"   Model: ε={self.model_loader.config.get('epsilon', 'unknown')}")
        print(f"   Accuracy: {self.model_loader.config.get('final_accuracy', 'unknown')}")
    
    def process_text(self, text: str, style: str = 'mask',
                    categories: Optional[List[PIICategory]] = None,
                    output_format: str = 'text') -> Dict:
        """
        Process text through the redaction pipeline.
        
        Args:
            text: Input text
            style: Redaction style
            categories: PII categories to redact
            output_format: 'text', 'json', or 'both'
        
        Returns:
            Dictionary with redaction results
        """
        print(f"📝 Processing text ({len(text)} characters)...")
        
        # Detect and redact
        redacted_text, entities = self.engine.redact_text(text, style, categories)
        
        # Prepare results
        results = {
            'original_text': text,
            'redacted_text': redacted_text,
            'detected_entities': [e.to_dict() for e in entities],
            'statistics': {
                'total_characters': len(text),
                'redacted_characters': sum(len(e.text) for e in entities),
                'entity_count': len(entities),
                'categories_detected': list(set(e.category.value for e in entities))
            },
            'pipeline_info': {
                'model_epsilon': self.model_loader.config.get('epsilon', 'unknown'),
                'model_accuracy': self.model_loader.config.get('final_accuracy', 'unknown'),
                'redaction_style': style
            }
        }
        
        print(f"✅ Detected {len(entities)} PII entities")
        
        # Return based on requested format
        if output_format == 'text':
            return redacted_text
        elif output_format == 'json':
            return results
        else:  # 'both'
            return results
    
    def process_file(self, input_path: Path, output_path: Optional[Path] = None,
                    style: str = 'mask', categories: Optional[List[PIICategory]] = None) -> Dict:
        """
        Process a text file through the redaction pipeline.
        
        Args:
            input_path: Path to input text file
            output_path: Path to save redacted text (None = auto-generate)
            style: Redaction style
            categories: PII categories to redact
        
        Returns:
            Dictionary with processing results
        """
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Process text
        results = self.process_text(text, style, categories, output_format='json')
        
        # Generate output path if not provided
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_redacted{input_path.suffix}"
        
        # Save redacted text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(results['redacted_text'])
        
        # Save detailed results as JSON
        results_path = output_path.parent / f"{input_path.stem}_redaction_details.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        results['output_files'] = {
            'redacted_text': str(output_path),
            'details_json': str(results_path)
        }
        
        print(f"💾 Redacted text saved to: {output_path}")
        print(f"💾 Detailed results saved to: {results_path}")
        
        return results
    
    def batch_process(self, input_dir: Path, output_dir: Optional[Path] = None,
                     style: str = 'mask', categories: Optional[List[PIICategory]] = None) -> Dict:
        """
        Process multiple text files in a directory.
        
        Args:
            input_dir: Directory containing text files
            output_dir: Directory to save redacted files (None = auto-generate)
            style: Redaction style
            categories: PII categories to redact
        
        Returns:
            Dictionary with batch processing results
        """
        if output_dir is None:
            output_dir = input_dir.parent / f"{input_dir.name}_redacted"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'files': []
        }
        
        # Process all .txt files in directory
        for file_path in input_dir.glob('*.txt'):
            try:
                print(f"\n📄 Processing: {file_path.name}")
                file_results = self.process_file(
                    file_path, 
                    output_dir / f"{file_path.stem}_redacted.txt",
                    style, 
                    categories
                )
                
                results['files'].append({
                    'input_file': str(file_path),
                    'output_file': file_results['output_files']['redacted_text'],
                    'entity_count': file_results['statistics']['entity_count']
                })
                results['processed_files'] += 1
                
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
                results['failed_files'] += 1
            
            results['total_files'] += 1
        
        # Save batch summary
        summary_path = output_dir / "batch_processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Batch processing complete:")
        print(f"   Total files: {results['total_files']}")
        print(f"   Processed: {results['processed_files']}")
        print(f"   Failed: {results['failed_files']}")
        print(f"   Summary saved to: {summary_path}")
        
        return results

# ========== COMMAND LINE INTERFACE ==========
def main():
    """Command line interface for the redaction pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PII Redaction Pipeline using Differential Privacy'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input text or path to text file/directory')
    parser.add_argument('--output', type=str, 
                       help='Output path for redacted text (default: auto-generated)')
    parser.add_argument('--model', type=str, 
                       default='outputs/models/final_epsilon_8.0',
                       help='Path to trained model')
    parser.add_argument('--style', type=str, default='mask',
                       choices=['mask', 'hash', 'partial', 'category_specific'],
                       help='Redaction style')
    parser.add_argument('--categories', type=str, nargs='+',
                       help='PII categories to redact (default: all)')
    parser.add_argument('--format', type=str, default='both',
                       choices=['text', 'json', 'both'],
                       help='Output format')
    parser.add_argument('--batch', action='store_true',
                       help='Process directory in batch mode')
    
    args = parser.parse_args()
    
    # Parse categories if provided
    categories = None
    if args.categories:
        categories = [PIICategory(cat.upper()) for cat in args.categories]
    
    # Initialize pipeline
    try:
        pipeline = RedactionPipeline(Path(args.model))
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return 1
    
    input_path = Path(args.input)
    
    if args.batch:
        # Batch processing mode
        if not input_path.is_dir():
            print(f"❌ Input must be a directory for batch processing: {args.input}")
            return 1
        
        results = pipeline.batch_process(
            input_path,
            Path(args.output) if args.output else None,
            args.style,
            categories
        )
        
    elif input_path.is_file():
        # Single file processing
        results = pipeline.process_file(
            input_path,
            Path(args.output) if args.output else None,
            args.style,
            categories
        )
        
        if args.format == 'text':
            print("\n" + "="*50)
            print("📄 REDACTED TEXT:")
            print("="*50)
            print(results['redacted_text'])
        elif args.format == 'json':
            print(json.dumps(results, indent=2))
        else:  # both
            print("\n" + "="*50)
            print("📄 REDACTED TEXT:")
            print("="*50)
            print(results['redacted_text'])
            print("\n" + "="*50)
            print("📊 DETECTION RESULTS:")
            print("="*50)
            print(json.dumps(results, indent=2))
    
    else:
        # Text input directly
        results = pipeline.process_text(
            args.input,
            args.style,
            categories,
            args.format
        )
        
        if args.format == 'text':
            print(results)
        elif args.format == 'json':
            print(json.dumps(results, indent=2))
        else:  # both
            print("\n" + "="*50)
            print("📄 REDACTED TEXT:")
            print("="*50)
            print(results['redacted_text'])
            print("\n" + "="*50)
            print("📊 DETECTION RESULTS:")
            print("="*50)
            print(json.dumps(results, indent=2))
    
    return 0

# ========== QUICK TEST FUNCTION ==========
def test_pipeline():
    """Test the redaction pipeline with sample text."""
    sample_text = """
    Dear John Doe,
    
    Please contact me at john.doe@example.com or call me at (123) 456-7890.
    My social security number is 123-45-6789 and my credit card is 4111 1111 1111 1111.
    I live at 123 Main Street, Springfield, and my IP address is 192.168.1.1.
    Let's meet on 12/31/2023.
    
    Best regards,
    Jane Smith
    CEO, Example Corp
    jane.smith@company.com
    """
    
    print("🧪 Testing Redaction Pipeline")
    print("="*60)
    
    # Initialize pipeline
    pipeline = RedactionPipeline()
    
    # Test different redaction styles
    styles = ['mask', 'hash', 'partial', 'category_specific']
    
    for style in styles:
        print(f"\n🔧 Testing style: {style}")
        print("-" * 40)
        
        results = pipeline.process_text(sample_text, style=style, output_format='json')
        
        print(f"Detected {results['statistics']['entity_count']} PII entities:")
        for entity in results['detected_entities']:
            print(f"  • {entity['category']}: {entity['text']} (confidence: {entity['confidence']:.2f})")
        
        print(f"\nRedacted text preview:")
        print(results['redacted_text'][:200] + "...")
    
    return pipeline

if __name__ == "__main__":
    # For testing: python -m src.redaction.redaction_pipeline
    import sys
    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        # Run test if no arguments
        test_pipeline()
EOF

chmod +x src/redaction/redaction_pipeline.py

# Create a simpler interface script
cat > redact_text.py << 'EOF'
#!/usr/bin/env python3
"""
Simple interface for the redaction pipeline.
"""
import sys
sys.path.insert(0, '.')

from src.redaction.redaction_pipeline import RedactionPipeline, PIICategory

def main():
    # Sample text with various PII
    sample_text = """Hello, my name is John Smith and my email is john.smith@example.com.
    You can call me at (555) 123-4567. My address is 123 Main St, Anytown, USA.
    My SSN is 123-45-6789 and my credit card is 4111-1111-1111-1111.
    I was born on 01/15/1985 and my IP address is 192.168.1.100."""
    
    print("🔐 PII Redaction Pipeline Demo")
    print("="*60)
    print("\n📝 Original Text:")
    print(sample_text)
    
    # Initialize pipeline with best model (ε=8.0)
    print("\n🚀 Initializing pipeline with best DP model (ε=8.0)...")
    pipeline = RedactionPipeline()
    
    # Test different redaction styles
    styles = [
        ('Full Masking', 'mask'),
        ('Hashing', 'hash'),
        ('Partial Redaction', 'partial'),
        ('Category Labels', 'category_specific')
    ]
    
    for style_name, style_code in styles:
        print(f"\n{'='*60}")
        print(f"🔧 Style: {style_name}")
        print(f"{'='*60}")
        
        result = pipeline.process_text(sample_text, style=style_code, output_format='json')
        
        print(f"\n📊 Detected {result['statistics']['entity_count']} PII entities:")
        for entity in result['detected_entities']:
            print(f"  • [{entity['category']}] {entity['text']} → {entity['confidence']:.1%} confidence")
        
        print(f"\n📄 Redacted Text:")
        print(result['redacted_text'])
    
    # Test specific category redaction
    print(f"\n{'='*60}")
    print("🎯 Redacting only EMAIL and PHONE:")
    print(f"{'='*60}")
    
    result = pipeline.process_text(
        sample_text, 
        style='category_specific',
        categories=[PIICategory.EMAIL, PIICategory.PHONE],
        output_format='json'
    )
    
    print(f"\n📄 Redacted Text (only emails and phones):")
    print(result['redacted_text'])
    
    print(f"\n{'='*60}")
    print("✅ Redaction Pipeline Ready!")
    print(f"{'='*60}")
    print("\nUsage examples:")
    print("  python redact_text.py --text 'Your text here'")
    print("  python src/redaction/redaction_pipeline.py --input document.txt")
    print("  python src/redaction/redaction_pipeline.py --input ./documents --batch")

if __name__ == "__main__":
    main()
EOF

chmod +x redact_text.py

# Create a command-line wrapper
cat > redact << 'EOF'
#!/bin/bash
# Wrapper script for the redaction pipeline

python src/redaction/redaction_pipeline.py "$@"
EOF

chmod +x redact

echo "✅ Created complete redaction pipeline!"