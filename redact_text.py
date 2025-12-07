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
