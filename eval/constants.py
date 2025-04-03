LIST_OPENAI_MODELS = ["auto", "gpt-4o-mini-2024-07-18"]

# Define patterns to detect thinking tokens
THINK_PATTERNS = [
    # HTML-style tags with permissive whitespace
    r'<think>\s*.*?\s*</think>',
    r'<thinking>\s*.*?\s*</thinking>',
    r'<thoughts>\s*.*?\s*</thoughts>',
    r'<thought>\s*.*?\s*</thought>',
    r'<Think>\s*.*?\s*</Think>',
    r'<Thinking>\s*.*?\s*</Thinking>',
    r'<Thoughts>\s*.*?\s*</Thoughts>',
    r'<Thought>\s*.*?\s*</Thought>',
    
    # Special separator style tags
    r'<\|begin_of_thought\|>\s*.*?\s*<\|end_of_thought\|>',
    r'<\|thinking\|>\s*.*?\s*<\|/thinking\|>',
    r'<\|thought\|>\s*.*?\s*<\|/thought\|>',
    r'<\|thoughts\|>\s*.*?\s*<\|/thoughts\|>',
    
    # Bracket style with permissive whitespace
    r'\[thinking\]\s*.*?\s*\[/thinking\]',
    r'\[thought\]\s*.*?\s*\[/thought\]',
    r'\[thoughts\]\s*.*?\s*\[/thoughts\]',
    r'\[THINKING\]\s*.*?\s*\[/THINKING\]',
    r'\[THOUGHT\]\s*.*?\s*\[/THOUGHT\]',
    r'\[THOUGHTS\]\s*.*?\s*\[/THOUGHTS\]',
    
    # Multiline variations with extra newlines
    r'<thinking>\s*.*?\s*</thinking>',
    r'<thought>\s*.*?\s*</thought>',
    r'<Think>\s*.*?\s*</Think>',
    r'<Thought>\s*.*?\s*</Thought>',
    
    # Comment-style tags with permissive whitespace
    r'<!--\s*thinking\s*-->\s*.*?\s*<!--\s*end thinking\s*-->',
    r'/\*\s*thinking\s*\*/\s*.*?\s*/\*\s*end thinking\s*\*/',
    
    # Extensive whitespace variations around tags
    r'<\s*thinking\s*>\s*.*?\s*<\s*/\s*thinking\s*>',
    r'<\s*thought\s*>\s*.*?\s*<\s*/\s*thought\s*>',
    r'<\s*Think\s*>\s*.*?\s*<\s*/\s*Think\s*>',
    r'<\s*Thought\s*>\s*.*?\s*<\s*/\s*Thought\s*>',
    
    # Edge cases with newlines and whitespace
    r'\n\s*<think>\s*.*?\s*</think>\s*\n?',
    r'\n\s*<thinking>\s*.*?\s*</thinking>\s*\n?',
    r'\n\s*<thought>\s*.*?\s*</thought>\s*\n?',
    r'\n\s*<thoughts>\s*.*?\s*</thoughts>\s*\n?',
    
    # Very permissive edge cases for broken or malformed tags
    r'<\s*thinking[^>]*>\s*.*?\s*(?:</\s*thinking\s*>|$)',
    r'<\s*thought[^>]*>\s*.*?\s*(?:</\s*thought\s*>|$)',
    r'\[\s*thinking[^\]]*\]\s*.*?\s*(?:\[\s*/\s*thinking\s*\]|$)',
    r'\[\s*thought[^\]]*\]\s*.*?\s*(?:\[\s*/\s*thought\s*\]|$)',
]