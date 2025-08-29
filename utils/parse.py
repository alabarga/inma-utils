#!/usr/bin/env python3
"""
Spanish Law Parser

This script reads Spanish law markdown files and generates structured JSON output.
The script parses hierarchical markdown structure with hash symbols (#) and extracts:
- Law title
- Publication date
- Preámbulo (preamble)
- Chapters with articles
- Disposiciones adicionales, transitorias, derogatorias, and finales
"""

import re
import json
import os
import uuid
from typing import Dict, List, Any, Optional
from pathlib import Path


class SpanishLawParser:
    def __init__(self):
        self.current_section = None
        self.current_article = None
        self.current_chapter = None
        
    def parse_markdown_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a Spanish law markdown file and return structured JSON."""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Initialize the structure
        law_data = {
            'ley': '',
            'publication': '',
            'preambulo': {'parrafos': []},
            'capitulos': [],
            'disposiciones adicionales': [],
            'disposiciones transitorias': [],
            'disposiciones derogatorias': [],
            'disposiciones finales': [],
            'stats': {}
        }
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Parse title
            if line == '# TITULO':
                i += 1
                if i < len(lines) and lines[i].startswith('## '):
                    law_data['ley'] = lines[i][3:].strip()
            
            # Parse publication
            elif line == '# PUBLICACION':
                i += 1
                if i < len(lines) and lines[i].startswith('## '):
                    law_data['publication'] = lines[i][3:].strip()
            
            # Parse preámbulo
            elif line == '# Preámbulo':
                i += 1
                parrafos = []
                while i < len(lines) and not lines[i].startswith('# '):
                    if lines[i].strip() and lines[i].startswith('## '):
                        # This is a numbered paragraph in the preamble
                        parrafo_text = lines[i][3:].strip()
                        if parrafo_text and not parrafo_text.isdigit():
                            parrafos.append({
                                'id': str(uuid.uuid4()),
                                'texto': parrafo_text
                            })
                    elif lines[i].strip() and not lines[i].startswith('##'):
                        # Regular paragraph text
                        parrafos.append({
                            'id': str(uuid.uuid4()),
                            'texto': lines[i].strip()
                        })
                    i += 1
                law_data['preambulo']['parrafos'] = [p for p in parrafos if p['texto']]
                continue
            
            # Parse chapters
            elif line.startswith('# CAPÍTULO'):
                chapter_info = self._parse_chapter(lines, i)
                if chapter_info:
                    law_data['capitulos'].append(chapter_info['chapter'])
                    i = chapter_info['next_index']
                    continue
            
            # Parse disposiciones adicionales
            elif (line.startswith('# DISPOSICIONES ADICIONALES') or 
                  'disposiciones adicionales' in line.lower() or
                  line.startswith('### Disposición Adicional')):
                disposiciones = self._parse_disposiciones(lines, i, 'adicionales')
                law_data['disposiciones adicionales'] = disposiciones['items']
                i = disposiciones['next_index']
                continue
            
            # Parse disposiciones transitorias
            elif (line.startswith('# DISPOSICIONES TRANSITORIAS') or 
                  'disposiciones transitorias' in line.lower() or
                  line.startswith('### Disposición Transitoria')):
                disposiciones = self._parse_disposiciones(lines, i, 'transitorias')
                law_data['disposiciones transitorias'] = disposiciones['items']
                i = disposiciones['next_index']
                continue
            
            # Parse disposiciones derogatorias
            elif (line.startswith('# DISPOSICIONES DEROGATORIAS') or 
                  'disposiciones derogatorias' in line.lower() or
                  line.startswith('### Disposición Derogatoria')):
                disposiciones = self._parse_disposiciones(lines, i, 'derogatorias')
                law_data['disposiciones derogatorias'] = disposiciones['items']
                i = disposiciones['next_index']
                continue
            
            # Parse disposiciones finales
            elif (line.startswith('# DISPOSICIONES FINALES') or 
                  'disposiciones finales' in line.lower() or
                  line.startswith('### Disposición Final')):
                disposiciones = self._parse_disposiciones(lines, i, 'finales')
                law_data['disposiciones finales'] = disposiciones['items']
                i = disposiciones['next_index']
                continue
            
            i += 1
        
        # Calculate stats
        law_data['stats'] = self._calculate_stats(law_data)
        
        return law_data
    
    def _calculate_stats(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics about paragraph lengths."""
        all_texts = []
        
        # Collect all texto fields from preambulo
        for parrafo in law_data['preambulo']['parrafos']:
            all_texts.append(parrafo['texto'])
        
        # Collect all texto fields from chapters
        for capitulo in law_data['capitulos']:
            for articulo in capitulo['articulos']:
                for parrafo in articulo['parrafos']:
                    all_texts.append(parrafo['texto'])
        
        # Collect all texto fields from disposiciones
        for disposicion in law_data['disposiciones adicionales']:
            if disposicion['texto']:
                all_texts.append(disposicion['texto'])
        
        for disposicion in law_data['disposiciones transitorias']:
            if disposicion['texto']:
                all_texts.append(disposicion['texto'])
        
        for disposicion in law_data['disposiciones derogatorias']:
            if disposicion['texto']:
                all_texts.append(disposicion['texto'])
        
        for disposicion in law_data['disposiciones finales']:
            if disposicion['texto']:
                all_texts.append(disposicion['texto'])
        
        # Calculate statistics
        if all_texts:
            lengths = [len(texto) for texto in all_texts]
            stats = {
                'total_paragraphs': len(all_texts),
                'max_length': max(lengths),
                'min_length': min(lengths),
                'avg_length': round(sum(lengths) / len(lengths), 2),
                'total_characters': sum(lengths)
            }
        else:
            stats = {
                'total_paragraphs': 0,
                'max_length': 0,
                'min_length': 0,
                'avg_length': 0,
                'total_characters': 0
            }
        
        return stats
    
    def _parse_chapter(self, lines: List[str], start_index: int) -> Optional[Dict[str, Any]]:
        """Parse a chapter and its articles."""
        chapter_line = lines[start_index].strip()
        
        # Extract chapter number and title
        chapter_match = re.search(r'CAPÍTULO\s+([IVX]+):\s*(.+)', chapter_line, re.IGNORECASE)
        if not chapter_match:
            return None
        
        chapter_num = chapter_match.group(1)
        chapter_title = chapter_match.group(2).strip()
        
        chapter = {
            'capitulo': chapter_num,
            'titulo': chapter_title,
            'articulos': []
        }
        
        i = start_index + 1
        current_article = None
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if we've reached the end of this chapter
            if (line.startswith('# CAPÍTULO') or 
                line.startswith('# DISPOSICIONES') or
                line.startswith('### Disposición')):
                break
            
            # Parse article
            if line.startswith('### Artículo'):
                if current_article:
                    chapter['articulos'].append(current_article)
                
                article_info = self._parse_article(lines, i)
                if article_info:
                    current_article = article_info['article']
                    i = article_info['next_index']
                    continue
            
            # Parse section (sección)
            elif line.startswith('## Sección') and current_article:
                section_text = line.replace('## ', '').strip()
                current_article['seccion'] = section_text
                i += 1
                continue
            
            i += 1
        
        # Add the last article if exists
        if current_article:
            chapter['articulos'].append(current_article)
        
        return {'chapter': chapter, 'next_index': i}
    
    def _parse_article(self, lines: List[str], start_index: int) -> Optional[Dict[str, Any]]:
        """Parse an article and its content."""
        article_line = lines[start_index].strip()
        
        # Extract article number and title
        article_match = re.search(r'Artículo\s+(\d+)\.\s*(.+)', article_line)
        if not article_match:
            return None
        
        article_num = int(article_match.group(1))
        article_title = article_match.group(2).strip()
        
        article = {
            'articulo': article_num,
            'titulo': article_title,
            'parrafos': []
        }
        
        i = start_index + 1
        current_parrafo = []
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if we've reached the end of this article
            if (line.startswith('### Artículo') or 
                line.startswith('# CAPÍTULO') or 
                line.startswith('# DISPOSICIONES') or
                line.startswith('## Sección') or
                line.startswith('### Disposición')):
                break
            
            # If line starts with ###, it's a new paragraph
            if line.startswith('### '):
                if current_parrafo:
                    article['parrafos'].append({
                        'id': str(uuid.uuid4()),
                        'texto': ' '.join(current_parrafo)
                    })
                    current_parrafo = []
                current_parrafo.append(line[4:].strip())
            elif line.startswith('#### '):
                if current_parrafo:
                    article['parrafos'].append({
                        'id': str(uuid.uuid4()),
                        'texto': ' '.join(current_parrafo)
                    })
                    current_parrafo = []
                current_parrafo.append(line[5:].strip())
            elif line and not line.startswith('#'):
                current_parrafo.append(line)
            
            i += 1
        
        # Add the last paragraph
        if current_parrafo:
            article['parrafos'].append({
                'id': str(uuid.uuid4()),
                'texto': ' '.join(current_parrafo)
            })
        
        return {'article': article, 'next_index': i}
    
    def _parse_disposiciones(self, lines: List[str], start_index: int, tipo: str) -> Dict[str, Any]:
        """Parse disposiciones (adicionales, transitorias, derogatorias, finales)."""
        items = []
        i = start_index + 1
        
        # If we're starting from a specific disposición line, process it
        if lines[start_index].startswith('### Disposición'):
            titulo = lines[start_index][4:].strip()
            texto = []
            
            j = start_index + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line.startswith('### ') or next_line.startswith('# '):
                    break
                if next_line:
                    texto.append(next_line)
                j += 1
            
            items.append({
                'titulo': titulo,
                'texto': ' '.join(texto) if texto else ''
            })
            
            i = j
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if we've reached the end of disposiciones
            if (line.startswith('# CAPÍTULO') or 
                (line.startswith('# DISPOSICIONES') and i != start_index) or
                (line.startswith('### Disposición') and not line.lower().startswith(f'disposición {tipo}'[:10]))):
                break
            
            # Parse individual disposición
            if line.startswith('### '):
                # Extract title and text
                titulo = line[4:].strip()
                texto = []
                
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if next_line.startswith('### ') or next_line.startswith('# '):
                        break
                    if next_line:
                        texto.append(next_line)
                    j += 1
                
                items.append({
                    'titulo': titulo,
                    'texto': ' '.join(texto) if texto else ''
                })
                
                i = j
                continue
            
            i += 1
        
        return {'items': items, 'next_index': i}


def main():
    """Main function to process all law files in the resources directory."""
    parser = SpanishLawParser()
    resources_dir = Path('resources')
    
    if not resources_dir.exists():
        print("Resources directory not found!")
        return
    
    # Process each markdown file
    for md_file in resources_dir.glob('*.md'):
        print(f"Processing {md_file.name}...")
        
        try:
            # Parse the law file
            law_data = parser.parse_markdown_file(str(md_file))
            
            # Generate output filename in resources directory
            output_file = resources_dir / f"{md_file.stem}.json"
            
            # Write JSON output
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(law_data, f, ensure_ascii=False, indent=2)
            
            print(f"Generated {output_file}")
            
        except Exception as e:
            print(f"Error processing {md_file.name}: {e}")


if __name__ == "__main__":
    main()
