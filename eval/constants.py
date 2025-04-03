LIST_OPENAI_MODELS = ["auto", "gpt-4o-mini-2024-07-18"]

# Define patterns to detect thinking tokens
THINK_PATTERNS = [
    # HTML-style tags
    r'<think>.*?</think>',
    r'<thinking>.*?</thinking>',
    r'<thoughts>.*?</thoughts>',
    r'<thought>.*?</thought>',
    r'<Think>.*?</Think>',
    r'<Thinking>.*?</Thinking>',
    r'<Thoughts>.*?</Thoughts>',
    r'<Thought>.*?</Thought>',
    
    # Special separator style tags
    r'<\|begin_of_thought\|>.*?<\|end_of_thought\|>',
    r'<\|thinking\|>.*?<\|/thinking\|>',
    r'<\|thought\|>.*?<\|/thought\|>',
    r'<\|thoughts\|>.*?<\|/thoughts\|>',
    
    # Bracket style
    r'\[thinking\].*?\[/thinking\]',
    r'\[thought\].*?\[/thought\]',
    r'\[thoughts\].*?\[/thoughts\]',
    r'\[THINKING\].*?\[/THINKING\]',
    r'\[THOUGHT\].*?\[/THOUGHT\]',
    r'\[THOUGHTS\].*?\[/THOUGHTS\]',
    
    # Multiline variations
    r'<thinking>\n?.*?\n?</thinking>',
    r'<thought>\n?.*?\n?</thought>',
    r'<Think>\n?.*?\n?</Think>',
    r'<Thought>\n?.*?\n?</Thought>',
    
    # Comment-style tags that some models use
    r'<!-- thinking -->.*?<!-- end thinking -->',
    r'/\* thinking \*/.*?/\* end thinking \*/',
    
    # Allow for whitespace around tags
    r'<\s*thinking\s*>.*?<\s*/\s*thinking\s*>',
    r'<\s*thought\s*>.*?<\s*/\s*thought\s*>',
]