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

    Args:
        text: The text to analyze
        patterns: List of regex patterns for thinking tokens

    Returns:
        List of (start, end) tuples indicating the span of each thinking block
    """
    spans = []
    
    # First pass: standard regex matching for simple tag pairs
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            spans.append(match.span())
    
    # Second pass: process more complex patterns
    # For each tag type, extract nested or unclosed blocks
    tag_pairs = [
        ("<think>", "</think>"),
        ("<thinking>", "</thinking>"),
        ("<thoughts>", "</thoughts>"),
        ("<thought>", "</thought>"),
        ("<Think>", "</Think>"),
        ("<Thinking>", "</Thinking>"),
        ("<Thoughts>", "</Thoughts>"),
        ("<Thought>", "</Thought>"),
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
        ("<!-- thinking -->", "<!-- end thinking -->"),
        ("/* thinking */", "/* end thinking */")
    ]
    
    # Add whitespace variations
    ws_pairs = [
        (r"<\s*thinking\s*>", r"<\s*/\s*thinking\s*>"),
        (r"<\s*thought\s*>", r"<\s*/\s*thought\s*>"),
    ]
    
    # Process all tag pairs
    for open_tag, close_tag in tag_pairs + ws_pairs:
        # Regex escape the tags if they're not already regexes (starting with r"\s")
        if not open_tag.startswith(r"\s") and not open_tag.startswith(r"<\s"):
            open_tag_pattern = re.escape(open_tag)
            close_tag_pattern = re.escape(close_tag)
        else:
            # Already regex patterns
            open_tag_pattern = open_tag
            close_tag_pattern = close_tag
            
        # Find all instances of open and close tags
        open_matches = list(re.finditer(open_tag_pattern, text))
        close_matches = list(re.finditer(close_tag_pattern, text))
        
        if not open_matches:
            continue
            
        # Case 1: Unclosed tag at the end
        if len(open_matches) > len(close_matches):
            # Find the last unmatched open tag
            last_open = None
            for i, open_match in enumerate(reversed(open_matches)):
                # Check if this open tag has a corresponding close tag
                has_close = False
                for close_match in close_matches:
                    if close_match.start() > open_match.end():
                        has_close = True
                        break
                if not has_close:
                    last_open = open_match
                    break
                    
            if last_open:
                # Add span from last open tag to end of string
                spans.append((last_open.start(), len(text)))
        
        # Case 2: Nested tags
        stack = []
        for i, char in enumerate(text):
            # Check for open tag at this position
            for open_match in open_matches:
                if open_match.start() == i:
                    stack.append(open_match.end())
                    break
                    
            # Check for close tag at this position
            for close_match in close_matches:
                if close_match.start() == i and stack:
                    # Found a close tag with a corresponding open tag on the stack
                    open_end = stack.pop()
                    # If stack is empty, we've closed a complete group
                    if not stack:
                        spans.append((open_match.start(), close_match.end()))
                    break
    
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
                
        return merged
    
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