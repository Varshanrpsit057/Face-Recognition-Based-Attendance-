import re

with open("app.py", "r", encoding="utf-8-sig") as f:
    content = f.read()

# Fix the broken width parameter
content = content.replace('width=" stretch\\)', 'use_container_width=True')

# Also check for any encoding issues with emojis
print(f"Remaining broken: {content.count('stretch')}")
print(f"use_container_width count: {content.count('use_container_width')}")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed!")
