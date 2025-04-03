"""
Utilities for post-processing reasoning blocks in model outputs.
"""

import re
from typing import List, Tuple, Dict, Optional, Callable


def extract_thinking_blocks(text: str, patterns: List[str]) -> List[Tuple[int, int]]:
    """
    Extract thinking blocks from text using a flexible matching approach.

    This function identifies not just regular tag pairs but also:
    1. Nested tags (multiple opens before any close)
    2. Unclosed tags (open tags that continue until the end of the string)
    3. Multiple open/close pairs
    4. Tags with excessive whitespace
    5. Malformed or broken tags

    Args:
        text: The text to analyze
        patterns: List of regex patterns for thinking tokens

    Returns:
        List of (start, end) tuples indicating the span of each thinking block
    """
    spans = []
    
    # First pass: standard regex matching with DOTALL flag to handle newlines
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            spans.append(match.span())
    
    # Second pass: look for isolated opening tags with no closing tag
    # Define basic open tag patterns with flexible whitespace
    open_tag_patterns = [
        r'<\s*think(?:\s*|\s+[^>]*)?>', 
        r'<\s*thinking(?:\s*|\s+[^>]*)?>', 
        r'<\s*thoughts(?:\s*|\s+[^>]*)?>', 
        r'<\s*thought(?:\s*|\s+[^>]*)?>', 
        r'<\s*Think(?:\s*|\s+[^>]*)?>', 
        r'<\s*Thinking(?:\s*|\s+[^>]*)?>', 
        r'<\s*Thoughts(?:\s*|\s+[^>]*)?>', 
        r'<\s*Thought(?:\s*|\s+[^>]*)?>', 
        r'\[\s*thinking(?:\s*|\s+[^\]]*)?]', 
        r'\[\s*thought(?:\s*|\s+[^\]]*)?]', 
        r'\[\s*thoughts(?:\s*|\s+[^\]]*)?]', 
        r'\[\s*THINKING(?:\s*|\s+[^\]]*)?]', 
        r'\[\s*THOUGHT(?:\s*|\s+[^\]]*)?]', 
        r'\[\s*THOUGHTS(?:\s*|\s+[^\]]*)?]',
        r'<\|thinking\|>', 
        r'<\|thought\|>', 
        r'<\|thoughts\|>',
        r'<\|begin_of_thought\|>'
    ]
    
    # Check for opening tags without corresponding closing tags
    for open_pattern in open_tag_patterns:
        open_matches = list(re.finditer(open_pattern, text, re.DOTALL))
        for open_match in open_matches:
            # Check if this opening tag already has a span (from the first pass)
            is_covered = False
            for start, end in spans:
                if open_match.start() >= start and open_match.start() < end:
                    is_covered = True
                    break
            
            # If not already covered, consider it an unclosed tag and add a span to the end
            if not is_covered:
                spans.append((open_match.start(), len(text)))
    
    # Third pass: process more complex patterns with nested tags and explicit tag pairs
    # For each tag type, extract nested or unclosed blocks
    tag_pairs = [
        ("<think", "</think>"),
        ("<thinking", "</thinking>"),
        ("<thoughts", "</thoughts>"),
        ("<thought", "</thought>"),
        ("<Think", "</Think>"),
        ("<Thinking", "</Thinking>"),
        ("<Thoughts", "</Thoughts>"),
        ("<Thought", "</Thought>"),
        ("<|begin_of_thought|>", "<|end_of_thought|>"),
        ("<|thinking|>", "<|/thinking|>"),
        ("<|thought|>", "<|/thought|>"),
        ("<|thoughts|>", "<|/thoughts|>"),
        ("[thinking]", "[/thinking]"),
        ("[thought]", "[/thought]"),
        ("[thoughts]", "[/thoughts]"),
        ("[THINKING]", "[/THINKING]"),
        ("[THOUGHT]", "[/THOUGHT]"),
        ("[THOUGHTS]", "[/THOUGHTS]"),
        ("<!-- thinking", "<!-- end thinking"),
        ("/* thinking", "/* end thinking")
    ]
    
    # Process all tag pairs with stack-based approach
    for open_tag_base, close_tag_base in tag_pairs:
        # Find all instances of open and close tags with flexible whitespace
        # This handles cases like "< thinking >" with lots of whitespace
        open_tag_pattern = f"{re.escape(open_tag_base)}\\s*[^><\\[\\]]*?>|{re.escape(open_tag_base)}\\s*[^><\\[\\]]*?\\]"
        close_tag_pattern = re.escape(close_tag_base)
        
        open_matches = list(re.finditer(open_tag_pattern, text, re.DOTALL))
        close_matches = list(re.finditer(close_tag_pattern, text, re.DOTALL))
        
        if not open_matches:
            continue
        
        # Sort by position
        open_positions = [(match.start(), "open", match) for match in open_matches]
        close_positions = [(match.start(), "close", match) for match in close_matches]
        all_positions = sorted(open_positions + close_positions)
        
        # Process nested tags with a stack
        stack = []
        for pos, tag_type, match in all_positions:
            if tag_type == "open":
                stack.append(match)
            elif tag_type == "close" and stack:
                open_match = stack.pop()
                # If stack is empty, we've closed a complete group
                if not stack:
                    spans.append((open_match.start(), match.end()))
        
        # Process leftover unclosed tags
        while stack:
            open_match = stack.pop()
            spans.append((open_match.start(), len(text)))
    
    # Merge overlapping spans
    if spans:
        spans.sort(key=lambda x: x[0])
        merged = [spans[0]]
        
        for current in spans[1:]:
            previous = merged[-1]
            # If current span overlaps with previous span
            if current[0] <= previous[1]:
                # Merge them
                merged[-1] = (previous[0], max(previous[1], current[1]))
            else:
                # Add as separate span
                merged.append(current)
                
        # Final check: ensure we don't have tiny spans (likely false positives)
        final_spans = []
        for start, end in merged:
            # If the span is reasonable in size (not just a tag name)
            if end - start >= 10:  # Minimum size to avoid false positives
                final_spans.append((start, end))
                
        return final_spans
    
    return spans


def process_thinking_blocks(text: str, patterns: List[str], 
                           process_fn: Callable[[str], str]) -> str:
    """
    Process thinking blocks in the text using the provided function.
    
    Args:
        text: The text containing thinking blocks
        patterns: List of regex patterns for thinking tokens
        process_fn: Function that takes a thinking block and returns processed text
        
    Returns:
        Text with processed thinking blocks
    """
    # Extract all thinking blocks
    spans = extract_thinking_blocks(text, patterns)
    
    # Process spans from end to beginning to avoid affecting positions
    result = text
    for start, end in sorted(spans, key=lambda x: x[0], reverse=True):
        # Extract the thinking block
        thinking_block = text[start:end]
        
        # Skip very short blocks (likely false positives)
        if len(thinking_block) < 20:
            continue
            
        # Process the thinking block
        processed_block = process_fn(thinking_block)
        
        # Replace in the result
        result = result[:start] + processed_block + result[end:]
    
    return result