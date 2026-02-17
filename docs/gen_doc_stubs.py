# gen_doc_stubs.py
from pathlib import Path

import mkdocs_gen_files

this_dir = Path(__file__).parent

src_root = (this_dir / "../circulus").resolve()

for path in src_root.rglob("*.py"):
    rel_path = path.relative_to(src_root)

    parts = tuple(rel_path.with_suffix("").parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = rel_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue
    else:
        doc_path = rel_path.with_suffix(".md")

    full_parts = ("circulus",) + parts
    identifier = ".".join(full_parts)

    output_filename = Path("references") / doc_path

    with mkdocs_gen_files.open(output_filename, "w") as fd:
        fd.write(f"::: {identifier}")

    mkdocs_gen_files.set_edit_path(output_filename, path)
