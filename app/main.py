import argparse
from pathlib import Path

from user import get_user_input
from vectorize import get_embedding, get_chunk_embedding
from db import get_connection, init_db, insert_item, insert_items, search
from reader import read_pdf, chunk_text

BATCH_SIZE = 32  # chunks per Ollama call


def load_pdf(conn, path: str) -> None:
    print("Path is: ", path)
    if not Path(path).is_file():
        print(f"File not found: {path}")
        return

    text = read_pdf(path)
    if not text.strip():
        print("No text extracted. The PDF may be scanned images (needs OCR).")
        return

    chunks = chunk_text(text)
    print(f"Extracted {len(text.split())} words -> {len(chunks)} chunks")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        # already [(chunk, vector), ...]
        pairs = get_chunk_embedding(batch)
        insert_items(conn, pairs)                # no zip
        print(f"  Stored {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")
    print("Done.")


def interactive(conn) -> None:
    print("Type text to store it. Prefix with '?' to search. 'quit' to exit.\n")

    while True:
        try:
            text = get_user_input()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text.lower() in {"quit", "exit", "q"}:
            break

        if text.startswith("?"):
            query = text[1:].strip()
            if not query:
                continue
            results = search(conn, get_embedding(query))
            if not results:
                print("  (no results, the table is empty)\n")
                continue
            for content, sim in results:
                print(f"  {sim:.3f}  {content[:120]}")
            print()
            continue

        vector = get_embedding(text)
        item_id = insert_item(conn, text, vector)
        print(f"  Stored id={item_id} ({len(vector)} dims)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create and search embeddings.")
    parser.add_argument("type", nargs="?", choices=[
                        "pdf"], help="type of file to load")
    parser.add_argument("path", nargs="?", help="path to the file")
    args = parser.parse_args()

    if args.type and not args.path:
        parser.error(f"'{args.type}' needs a file path")

    conn = get_connection()
    init_db(conn)

    try:
        if args.type == "pdf":
            load_pdf(conn, args.path)
        else:
            interactive(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
