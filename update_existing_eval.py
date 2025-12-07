import sys

# Read the existing evaluate_dp_model.py
with open('src/evaluation/evaluate_dp_model.py', 'r') as f:
    content = f.read()

# Add baseline comparison option to the argument parser
if '--compare_all' in content:
    # Find the argument parser section and add --with_baseline option
    new_arg = "    parser.add_argument('--with_baseline', action='store_true', help='Include baseline in comparison')\n"
    
    # Insert after --compare_all line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '--compare_all' in line and 'parser.add_argument' in line:
            lines.insert(i + 1, new_arg)
            break
    
    content = '\n'.join(lines)
    
    # Also update the compare_multiple_models function to accept with_baseline parameter
    if 'def compare_multiple_models' in content:
        # Update function signature
        content = content.replace(
            'def compare_multiple_models(epsilons=[8.0, 5.0, 3.0, 2.0, 1.0, 0.5], output_dir=None):',
            'def compare_multiple_models(epsilons=[8.0, 5.0, 3.0, 2.0, 1.0, 0.5], output_dir=None, with_baseline=False):'
        )
        
        # Find the function call in main() and add with_baseline parameter
        if 'compare_multiple_models(args.epsilons, output_dir)' in content:
            content = content.replace(
                'compare_multiple_models(args.epsilons, output_dir)',
                'compare_multiple_models(args.epsilons, output_dir, args.with_baseline)'
            )

with open('src/evaluation/evaluate_dp_model.py', 'w') as f:
    f.write(content)

print("✅ Updated evaluate_dp_model.py to include --with_baseline option")
