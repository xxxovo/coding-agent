import os
import re

MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".memory")
INDEX_FILE = os.path.join(MEMORY_DIR, "MEMORY.md")

VALID_TYPES = {"user", "feedback", "project", "reference"}

def init_memory_dir():
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR)
    if not os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            f.write("# Memory Index\n\n")

def read_index() -> str:
    """Read the index file. If it doesn't exist, return empty."""
    init_memory_dir()
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def update_index():
    """Rebuild MEMORY.md from all memory files to enforce 200 lines / 25KB"""
    init_memory_dir()
    lines = ["# Memory Index\n\n"]
    
    # Read all markdown files (except MEMORY.md)
    files = [f for f in os.listdir(MEMORY_DIR) if f.endswith(".md") and f != "MEMORY.md"]
    
    entries = []
    for fl in files:
        filepath = os.path.join(MEMORY_DIR, fl)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                # parse frontmatter
                match_name = re.search(r"name:\s*(.+)", content)
                match_desc = re.search(r"description:\s*(.+)", content)
                
                name = match_name.group(1).strip() if match_name else fl.replace(".md", "")
                desc = match_desc.group(1).strip() if match_desc else "No description"
                
                entries.append(f"- [{name}]({fl}) -- {desc}")
        except Exception:
            continue
            
    # Constraints: Max 200 lines, Max 25 KB
    MAX_LINES = 200
    MAX_BYTES = 25000
    
    for entry in entries:
        if len(lines) >= MAX_LINES:
            lines.append("... (Index truncated due to 200 lines limit) ...")
            break
        # truncate single entry to 150 chars max for the description
        if len(entry) > 150:
            entry = entry[:147] + "..."
        lines.append(entry + "\n")
        
    final_content = "".join(lines)
    if len(final_content.encode("utf-8")) > MAX_BYTES:
        # crude truncation
        final_content = final_content.encode("utf-8")[:MAX_BYTES].decode("utf-8", "ignore")
        final_content += "\n... (Index truncated due to 25KB size limit) ..."
        
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        f.write(final_content)

def save_memory(name: str, description: str, memory_type: str, content: str) -> str:
    """Save a memory as a markdown file with YAML frontmatter."""
    init_memory_dir()
    if memory_type not in VALID_TYPES:
        return f"Error: memory_type must be one of {VALID_TYPES}"
        
    # sanitize filename
    safe_name = re.sub(r'[\W_]+', '-', name.lower())
    if not safe_name:
        safe_name = "memory"
    filename = f"{safe_name}.md"
    filepath = os.path.join(MEMORY_DIR, filename)
    
    file_content = f"""---
name: {name}
description: {description}
type: {memory_type}
---

{content}
"""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(file_content)
        update_index()
        return f"Memory successfully saved to {filename} and index updated."
    except Exception as e:
        return f"Error saving memory: {str(e)}"

def read_memory(filename: str) -> str:
    """Read the detailed content of a specific memory file."""
    filepath = os.path.join(MEMORY_DIR, filename)
    if not os.path.exists(filepath):
        return f"Error: Memory file {filename} not found."
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading memory: {str(e)}"
