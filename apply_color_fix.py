import sys

with open('src/training/train_dp_final_working.py', 'r') as f:
    lines = f.readlines()

# Fix line by line
for i, line in enumerate(lines):
    if "'purple-o'" in line:
        lines[i] = line.replace("'purple-o'", "'purple', marker='o'")
    if "'orange-o'" in line:
        lines[i] = line.replace("'orange-o'", "'orange', marker='o'")
    if "'brown-s'" in line:
        lines[i] = line.replace("'brown-s'", "'brown', marker='s'")
    if "'green-o'" in line:
        lines[i] = line.replace("'green-o'", "'green', marker='o'")

with open('src/training/train_dp_final_working.py', 'w') as f:
    f.writelines(lines)

print("✅ Applied comprehensive fix")
