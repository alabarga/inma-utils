#!/usr/bin/env python3
"""
Script to clean law documents for RAG system ingestion.

This script removes images, bold markup, spans, and other HTML/markdown elements
while preserving the hierarchical structure of the legal document.
"""

import re
import sys
from pathlib import Path


def clean_law_document(input_file_path, output_file_path=None):
    """
    Clean a law document by removing unwanted markup and formatting.
    
    Args:
        input_file_path (str): Path to the input markdown file
        output_file_path (str): Path to the output cleaned file (optional)
    
    Returns:
        str: Cleaned content
    """
    
    # Read the input file
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File {input_file_path} not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    # Apply cleaning rules
    cleaned_content = apply_cleaning_rules(content)
    
    # Write to output file if specified
    if output_file_path:
        try:
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_content)
            print(f"Cleaned content written to: {output_file_path}")
        except Exception as e:
            print(f"Error writing output file: {e}")
            return None
    
    return cleaned_content


def apply_cleaning_rules(content):
    """
    Apply all cleaning rules to the content.
    
    Args:
        content (str): Original content
        
    Returns:
        str: Cleaned content
    """
    
    # 1. Remove images (e.g., ![](_page_4_Picture_9.jpeg))
    content = re.sub(r'!\[\]\([^)]*\)', '', content)
    
    # 2. Remove bold markup ** ** and italic markup * *
    content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)
    content = re.sub(r'\*(.*?)\*', r'\1', content)
    
    # 3. Remove spans and other HTML/XML markup
    # Remove span tags with IDs like <span id="page-7-1">
    content = re.sub(r'<span[^>]*>', '', content)
    content = re.sub(r'</span>', '', content)
    
    # Remove other HTML/XML tags
    content = re.sub(r'<[^>]+>', '', content)
    
    # 4. Replace <sup>&</sup>quot; with " and &quot; with "
    content = re.sub(r'<sup>&</sup>quot;', '"', content)
    content = re.sub(r'&quot;', '"', content)
    
    # 5. Fix broken CAPÍTULO words (CAP ULO -> CAPÍTULO)
    content = re.sub(r'CAP\s+ULO', 'CAPÍTULO', content)
    content = re.sub(r'CAP\s+ÍTULO', 'CAPÍTULO', content)
    
    # 6. Standardize hierarchy using # for headings
    content = standardize_hierarchy(content)
    
    # 7. Clean up extra whitespace and empty lines
    content = clean_whitespace(content)
    
    return content


def standardize_hierarchy(content):
    """
    Standardize the document hierarchy:
    # CAPITULO
    ## SECCION  
    ### ARTICULO
    #### PARRAFO
    
    Args:
        content (str): Content to standardize
        
    Returns:
        str: Content with standardized hierarchy
    """
    
    # Split content into lines for processing
    lines = content.split('\n')
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines for now, we'll handle them later
        if not line:
            processed_lines.append(line)
            continue
            
        # Handle different hierarchy levels
        processed_line = process_hierarchy_line(line)
        processed_lines.append(processed_line)
    
    return '\n'.join(processed_lines)


def process_hierarchy_line(line):
    """
    Process a single line to apply hierarchy rules.
    
    Args:
        line (str): Line to process
        
    Returns:
        str: Processed line with correct hierarchy
    """
    
    # Remove existing markdown headers to start fresh
    line = re.sub(r'^#+\s*', '', line)
    
    # Don't add hashtags to simple list items (a), b), 1), etc.
    if re.search(r'^\s*[a-z]\)\s+', line, re.IGNORECASE) or re.search(r'^\s*\d+\)\s+', line):
        return line
    
    # Don't add hashtags to sub-items like a.1), b.2), etc.
    if re.search(r'^\s*[a-z]\.\d+\)\s+', line, re.IGNORECASE):
        return line
    
    # CAPITULO level (# CAPITULO)
    if re.search(r'CAPÍTULO|CAPITULO', line, re.IGNORECASE):
        return f"# {line}"
    
    # SECCION level (## SECCION)
    if re.search(r'Sección|SECCION|Sección\s+\d+', line, re.IGNORECASE):
        return f"## {line}"
    
    # ARTICULO level (### ARTICULO)
    if re.search(r'Artículo\s+\d+|ARTICULO\s+\d+', line, re.IGNORECASE):
        return f"### {line}"
    
    # Special cases for numbered paragraphs (#### PARRAFO) - but only if they don't look like list items
    if re.search(r'^\d+\.\s+', line) and not re.search(r'^\d+\.\s*[a-z]\)', line, re.IGNORECASE):
        return f"#### {line}"
    
    # Keep titles and other important headings
    if re.search(r'^(TITULO|PUBLICACION|Preámbulo|decreto:|Disposición)', line, re.IGNORECASE):
        return f"# {line}"
    
    # Keep disposition sections
    if re.search(r'Disposición\s+(Adicional|Transitoria|Derogatoria|Final)', line, re.IGNORECASE):
        return f"## {line}"
    
    # Return line as-is if no hierarchy pattern matches
    return line


def clean_whitespace(content):
    """
    Clean up extra whitespace and normalize line breaks.
    
    Args:
        content (str): Content to clean
        
    Returns:
        str: Content with cleaned whitespace
    """
    
    # Remove multiple consecutive blank lines
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    # Remove trailing whitespace from lines
    lines = content.split('\n')
    cleaned_lines = [line.rstrip() for line in lines]
    
    # Join lines back together
    content = '\n'.join(cleaned_lines)
    
    # Remove leading and trailing whitespace from the entire content
    content = content.strip()
    
    return content


def main():
    """Main function to handle command line arguments."""
    
    if len(sys.argv) < 2:
        print("Usage: python clean_law_document.py <input_file> [output_file]")
        print("Example: python clean_law_document.py 12302.md 12302_cleaned.md")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # If no output file specified, create one based on input file name
    if not output_file:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}")
    
    print(f"Cleaning document: {input_file}")
    print(f"Output will be saved to: {output_file}")
    
    result = clean_law_document(input_file, output_file)
    
    if result:
        print("Document cleaning completed successfully!")
        print(f"Original file size: {len(open(input_file, 'r', encoding='utf-8').read())} characters")
        print(f"Cleaned file size: {len(result)} characters")
    else:
        print("Document cleaning failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
