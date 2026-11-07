from user import get_user_input
from vectorize import get_embedding
from db import get_connection, init_db, insert_item, search


def main():
    # Setup the Database
    conn = get_connection()
    init_db(conn)

    print("Type text to store it. Prefix with '?' to search. 'quit' to exit.\n")

    # Start loop for embedding
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

        # Search mode: "? run an LLM locally"
        if text.startswith("?"):
            query = text[1:].strip()
            if not query:
                continue
            vector = get_embedding(query)
            results = search(conn, vector)
            if not results:
                print("  (no results, the table is empty)\n")
                continue
            for content, sim in results:
                print(f"  {sim:.3f}  {content}")
            print()
            continue

        # Store mode
        vector = get_embedding(text)
        item_id = insert_item(conn, text, vector)
        print(f"  Stored id={item_id} ({len(vector)} dims)\n")

    conn.close()


if __name__ == "__main__":
    main()
