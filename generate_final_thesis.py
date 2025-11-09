"""
Generate Final Thesis Markdown from Modified JSONL
"""

import json
from pathlib import Path


def generate_final_thesis():
    """Generate final thesis markdown from modified sections"""

    input_file = Path("modified_thesis_sections.jsonl")
    output_file = Path("final_thesis.md")

    if not input_file.exists():
        print("[ERROR] modified_thesis_sections.jsonl not found!")
        return

    print("Loading modified thesis sections...")
    sections = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            sections.append(json.loads(line))

    print(f"[OK] Loaded {len(sections)} sections")

    # Generate markdown
    print("\nGenerating final thesis markdown...")

    markdown_content = []

    # Title page
    markdown_content.append("# Automated Shoulder Implant Manufacturer Classification Using Deep Learning")
    markdown_content.append("\n## Master's Thesis")
    markdown_content.append("\n**AI-Powered Medical Image Analysis for Orthopedic Applications**")
    markdown_content.append("\n---\n")

    # Add each section
    for i, section in enumerate(sections):
        header = section['header']
        content = section['content']

        # Add section header
        markdown_content.append(f"\n# {header}\n")

        # Add content
        markdown_content.append(content)

        # Add page break after each major section
        markdown_content.append("\n\n---\n")

    # Combine all content
    final_markdown = '\n'.join(markdown_content)

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_markdown)

    print(f"\n[OK] Generated: {output_file}")
    print(f"  Total length: {len(final_markdown):,} characters")
    print(f"  Total sections: {len(sections)}")

    # Statistics
    total_words = len(final_markdown.split())
    print(f"  Estimated word count: {total_words:,} words")
    print(f"  Estimated pages (250 words/page): {total_words // 250} pages")

    print("\n[OK] Final thesis ready!")


if __name__ == "__main__":
    generate_final_thesis()
