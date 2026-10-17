from pathlib import Path

INPUT_PATH = Path("extracted_text.txt")
OUTPUT_PATH = Path("chunks.txt")


def split_into_chunks(
    text: str,
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[str]:
    """Split text into overlapping chunks of whole words."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH.resolve()}"
        )

    text = INPUT_PATH.read_text(encoding="utf-8")
    chunks = split_into_chunks(text)

    output = "\n\n".join(
        f"--- Chunk {i} ---\n{chunk}"
        for i, chunk in enumerate(chunks, start=1)
    )

    OUTPUT_PATH.write_text(output, encoding="utf-8")

    print(f"Total characters: {len(text)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Saved chunks to: {OUTPUT_PATH.resolve()}")

    if chunks:
        print("\nFirst chunk preview:\n")
        print(chunks[0][:500])


if __name__ == "__main__":
    main()