"""
PDF Parser - Convert PDFs to Markdown and save as JSONL
Uses pymupdf4llm for best markdown conversion
"""

import os
import json
import logging
from pathlib import Path
import pymupdf4llm

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def parse_pdf_to_markdown(pdf_path: Path) -> dict:
    """Parse PDF to markdown using pymupdf4llm"""
    try:
        # Convert PDF to markdown
        md_text = pymupdf4llm.to_markdown(str(pdf_path))

        return {
            'filename': pdf_path.name,
            'title': pdf_path.stem,
            'content': md_text,
            'status': 'success'
        }
    except Exception as e:
        logger.warning(f"✗ Failed to parse {pdf_path.name}: {e}")
        return {
            'filename': pdf_path.name,
            'title': pdf_path.stem,
            'content': '',
            'status': 'failed',
            'error': str(e)
        }


def main():
    logger.info("=== PDF to Markdown Parser ===\n")

    # Setup paths
    pdf_dir = Path("downloaded_papers")
    output_file = "parsed_papers.jsonl"

    # Get all PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files\n")

    if not pdf_files:
        logger.warning("No PDFs found in downloaded_papers/")
        return

    # Parse each PDF
    parsed_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"[{i}/{len(pdf_files)}] Parsing: {pdf_path.name}")

            result = parse_pdf_to_markdown(pdf_path)

            if result['status'] == 'success':
                logger.info(f"  ✓ Converted {len(result['content'])} chars")
                parsed_count += 1

            # Write to JSONL
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    logger.info(f"\n✓ Parsed {parsed_count}/{len(pdf_files)} PDFs")
    logger.info(f"✓ Saved to {output_file}")


if __name__ == "__main__":
    main()
