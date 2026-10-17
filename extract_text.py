from pathlib import Path
import pymupdf

PDF_PATH = Path("notes.pdf")
OUTPUT_PATH = Path("extracted_text.txt")


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract readable text from every page of a PDF."""

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path.resolve()}"
        )

    with pymupdf.open(pdf_path) as document:
        pages = []

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True)

            pages.append(
                f"--- Page {page_number} ---\n{text}"
            )

    return "\n\n".join(pages)


def main():
    try:
        extracted_text = extract_pdf_text(PDF_PATH)

        OUTPUT_PATH.write_text(
            extracted_text,
            encoding="utf-8"
        )

        print("PDF text extracted successfully!")
        print(f"Saved to: {OUTPUT_PATH.resolve()}")
        print(f"Characters extracted: {len(extracted_text)}")
        print("\nPreview:\n")
        print(extracted_text[:1500])

        if not extracted_text.strip():
            print(
                "\nWarning: No text was found. "
                "The PDF may contain scanned pages."
            )

    except (FileNotFoundError, pymupdf.FileDataError) as error:
        print(f"Could not read the PDF: {error}")


if __name__ == "__main__":
    main()