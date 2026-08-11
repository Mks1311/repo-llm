from pathlib import Path
import ast


def chunk_by_ast(file_path, max_chunk_size=500):
    """
    Chunk a Python file using AST-based chunking (built-in `ast` module).

    Groups top-level statements (functions, classes, imports, etc.) into
    chunks up to ~max_chunk_size words. Oversized classes are split further
    into per-method chunks so no single chunk balloons in size.
    """

    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    if not source.strip():
        return []

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        print(f"  [WARN] Skipping AST chunking for {file_path}: SyntaxError: {e}")
        return []

    def get_segment(node):
        """Extract a node's original source text."""
        segment = ast.get_source_segment(source, node)
        if segment is not None:
            return segment
        # Fallback for nodes get_source_segment can't handle
        lines = source.splitlines(keepends=True)
        start = node.lineno - 1
        end = getattr(node, "end_lineno", node.lineno)
        return "".join(lines[start:end])

    def node_word_count(node):
        return len(get_segment(node).split())

    def make_chunk(nodes, class_name=None):
        texts = [get_segment(n) for n in nodes]
        start_line = nodes[0].lineno
        end_line = getattr(nodes[-1], "end_lineno", nodes[-1].lineno)
        metadata = {
            "start_line_no": start_line - 1,
            "end_line_no": end_line - 1,
            "node_types": [type(n).__name__ for n in nodes],
        }
        text = "\n\n".join(texts)
        if class_name:
            metadata["class_name"] = class_name
            text = f"class {class_name}:\n" + text

        return {
            "text": text,
            "file_path": str(file_path),
            "start_line": start_line,
            "end_line": end_line,
            "type": "python",
            "metadata": metadata,
        }

    def split_class(class_node):
        """Split an oversized class into grouped per-method chunks."""
        sub_chunks = []
        group, word_count = [], 0

        for node in class_node.body:
            words = node_word_count(node)

            if group and word_count + words > max_chunk_size:
                sub_chunks.append(make_chunk(group, class_name=class_node.name))
                group, word_count = [], 0

            group.append(node)
            word_count += words

        if group:
            sub_chunks.append(make_chunk(group, class_name=class_node.name))

        return sub_chunks

    chunks = []
    group, word_count = [], 0

    for node in tree.body:
        words = node_word_count(node)

        # Oversized single node (e.g. a huge class) — flush pending group,
        # then handle it on its own.
        if words > max_chunk_size:
            if group:
                chunks.append(make_chunk(group))
                group, word_count = [], 0

            if isinstance(node, ast.ClassDef):
                chunks.extend(split_class(node))
            else:
                chunks.append(make_chunk([node]))  # can't split a function further
            continue

        if group and word_count + words > max_chunk_size:
            chunks.append(make_chunk(group))
            group, word_count = [], 0

        group.append(node)
        word_count += words

    if group:
        chunks.append(make_chunk(group))

    return chunks
