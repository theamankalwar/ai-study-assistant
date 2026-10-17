
from sentence_transformers import SentenceTransformer

from search_chunks import (
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    load_chunks,
    retrieve_chunks,
)


def main():
    chunks = load_chunks(CHUNKS_PATH)

    if not chunks:
        print("No chunks found. Check chunks.txt.")
        return

    print(f"Loaded {len(chunks)} chunks.")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Creating embeddings...")
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    # Change these questions to match topics in your PDF.
    questions = [
        "What does the ALU do?",
        "Explain the architecture of the 8085.",
        "What is the function of the accumulator?",
        "How does the microprocessor handle interrupts?",
        "Explain the different addressing modes.",
    ]

    for question in questions:
        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        results = retrieve_chunks(
            question,
            chunks,
            embeddings,
            model,
            top_k=3,
        )

        for index, score, chunk in results:
            print(f"\nChunk {index + 1} | Score: {score:.3f}")
            print(chunk[:500])
            print("-" * 50)


if __name__ == "__main__":
    main()
