"""
Thesis Loader - Notebook Friendly Script
Load thesis.md and split by headers, plus load associated images
"""

import os
from pathlib import Path
from PIL import Image
import re
import json

# Configuration
THESIS_FOLDER = "thesis"
THESIS_FILE = "Thesis.md"
IMAGE_FILES = [
    "ConfusionMatrix.png",
    "diagram_1.png",
    "diagram_2.png",
    "loss_accuracy_precision&recall.png"
]


def load_thesis(thesis_path):
    """
    Load thesis markdown file and return its content

    Args:
        thesis_path: Path to the thesis.md file

    Returns:
        str: Full content of the thesis
    """
    with open(thesis_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def split_by_headers(content):
    """
    Split thesis content by main headers (single # only)
    Sub-headers (##, ###, etc.) are kept as part of the section content

    Args:
        content: Full thesis content as string

    Returns:
        list: List of dictionaries with 'title' and 'content' for each main section
    """
    # Split by lines
    lines = content.split('\n')

    sections = []
    current_section = None

    for line in lines:
        # Check if line is a MAIN header (single # only)
        header_match = re.match(r'^#\s+(.+)$', line)

        if header_match:
            # Save previous section if exists
            if current_section is not None:
                sections.append(current_section)

            # Start new section
            title = header_match.group(1).strip()
            current_section = {
                'title': title,
                'header_line': line,
                'content': []
            }
        else:
            # Add line to current section (including sub-headers)
            if current_section is not None:
                current_section['content'].append(line)
            else:
                # Content before first header
                if not sections:
                    sections.append({
                        'title': 'Preamble',
                        'header_line': '',
                        'content': [line]
                    })
                else:
                    sections[0]['content'].append(line)

    # Don't forget the last section
    if current_section is not None:
        sections.append(current_section)

    # Join content lines back into strings
    for section in sections:
        section['content'] = '\n'.join(section['content']).strip()

    return sections


def load_images(thesis_folder, image_files):
    """
    Load all images from the thesis folder

    Args:
        thesis_folder: Path to thesis folder
        image_files: List of image filenames

    Returns:
        dict: Dictionary mapping image names to PIL Image objects
    """
    images = {}

    for img_file in image_files:
        img_path = os.path.join(thesis_folder, img_file)
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                # Store with a friendly name (without extension)
                img_name = os.path.splitext(img_file)[0]
                images[img_name] = img
                print(f"✓ Loaded: {img_file} ({img.size[0]}x{img.size[1]})")
            except Exception as e:
                print(f"✗ Error loading {img_file}: {e}")
        else:
            print(f"✗ Not found: {img_file}")

    return images


def save_sections_to_jsonl(sections, output_path):
    """
    Save sections to JSONL format with metadata

    Args:
        sections: List of section dictionaries
        output_path: Path to output JSONL file

    Each line in the JSONL file contains:
    {
        "section_number": <int>,
        "header": <string>,
        "content": <string>
    }
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, section in enumerate(sections):
            jsonl_entry = {
                "section_number": i,
                "header": section['title'],
                "content": section['content']
            }
            f.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(sections)} sections to {output_path}")


def display_section_summary(sections):
    """
    Display a summary of all sections

    Args:
        sections: List of section dictionaries
    """
    print("\n" + "="*70)
    print("THESIS SECTIONS SUMMARY")
    print("="*70)

    for i, section in enumerate(sections):
        title = section['title']
        content_preview = section['content'][:100].replace('\n', ' ')

        print(f"\n[{i}] # {title}")
        print(f"    Content length: {len(section['content'])} chars")
        print(f"    Preview: {content_preview}...")


# Main execution block
if __name__ == "__main__":
    print("Loading Thesis and Images...\n")

    # Build paths
    thesis_path = os.path.join(THESIS_FOLDER, THESIS_FILE)

    # 1. Load thesis content
    print(f"Loading thesis from: {thesis_path}")
    thesis_content = load_thesis(thesis_path)
    print(f"✓ Loaded {len(thesis_content)} characters\n")

    # 2. Split by headers
    print("Splitting thesis by headers...")
    sections = split_by_headers(thesis_content)
    print(f"✓ Found {len(sections)} sections\n")

    # 3. Load images
    print("Loading images...")
    images = load_images(THESIS_FOLDER, IMAGE_FILES)
    print(f"\n✓ Loaded {len(images)} images")

    # 4. Display summary
    display_section_summary(sections)

    # 5. Save to JSONL
    print("\nSaving sections to JSONL...")
    jsonl_output_path = os.path.join(THESIS_FOLDER, "thesis_sections.jsonl")
    save_sections_to_jsonl(sections, jsonl_output_path)

    # Return key variables for notebook use
    print("\n" + "="*70)
    print("AVAILABLE VARIABLES:")
    print("="*70)
    print("- thesis_content : Full thesis text")
    print("- sections       : List of section dicts (title, content)")
    print("- images         : Dict of PIL Image objects")
    print(f"  Image keys: {list(images.keys())}")
    print("\nExample usage:")
    print("  sections[0]['title']  # Get first section title")
    print("  sections[0]['content']  # Get content (includes sub-headers)")
    print("  images['ConfusionMatrix'].show()  # Display confusion matrix")
    print("\nJSONL file saved to: thesis/thesis_sections.jsonl")
