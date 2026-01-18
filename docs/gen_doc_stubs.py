# gen_doc_stubs.py
import mkdocs_gen_files
from pathlib import Path

# 1. Setup paths
this_dir = Path(__file__).parent
# Assuming this script is in the root of your repo
src_root = (this_dir / "../circulus").resolve() 

# 2. Iterate over the Python files
for path in src_root.rglob("*.py"):
    
    # Get path relative to the package folder (e.g., 'utils.py' or 'sub/mod.py')
    rel_path = path.relative_to(src_root)
    
    # Convert to parts: ('utils',)
    parts = tuple(rel_path.with_suffix("").parts)

    # Handle __init__ files
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = rel_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue
    else:
        doc_path = rel_path.with_suffix(".md")

    # --- THE FIX FOR YOUR CRASH ---
    # We are inside the 'circulus' folder, so 'parts' lacks the package name.
    # We MUST prepend 'circulus' so the identifier becomes 'circulus.utils'
    # instead of just 'utils' (which fails) or '' (which crashes).
    full_parts = ("circulus",) + parts
    identifier = ".".join(full_parts)
    
    # Create the virtual markdown file
    # We use a string path relative to 'docs/'. 
    # e.g., "references/utils.md"
    output_filename = Path("references") / doc_path
    
    with mkdocs_gen_files.open(output_filename, "w") as fd:
        fd.write(f"::: {identifier}")

    # Set the edit path so the "Edit this page" button on GitHub works
    mkdocs_gen_files.set_edit_path(output_filename, path)