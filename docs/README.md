# EVAlchemy Chat Benchmark Response Formats

This document catalogs the different response formats returned by various chat benchmarks in the EVAlchemy framework. Understanding these formats is essential for implementing post-processing logic that works across all benchmarks.

## Response Format Overview

In the EVAlchemy framework, each benchmark's `generate_responses` method produces results that are then passed to the `evaluate_responses` method. The structure of these results varies between benchmarks.

### Common Response Format Patterns

Based on analysis of the codebase, we've identified these common patterns:

1. **MTBench Format**
   - Returns a dictionary with `model_id` key
   - Actual responses are stored in a separate JSONL file
   - Format structure:
     ```python
     {
         "model_id": "model_identifier_string",
     }
     ```

2. **Choice-based Format**
   - Results contain a `choices` array with response options
   - Each choice has an index and text content
   - Format structure:
     ```python
     {
         "choices": [
             {
                 "index": 0,
                 "text": "Model response text...",
                 "turns": ["First turn text", "Second turn text"]  # For multi-turn benchmarks
             },
             # Additional choices if present
         ]
     }
     ```

3. **Array-based Format**
   - Direct array of response strings
   - Format structure:
     ```python
     {
         "responses": ["Model response text 1", "Model response text 2", ...]
     }
     ```

4. **Question-Answer Format**
   - Maps question IDs to model responses
   - Format structure:
     ```python
     {
         "question_id_1": "Model response 1",
         "question_id_2": "Model response 2",
         ...
     }
     ```

## Benchmark-Specific Formats

### MTBench

MTBench follows a unique pattern where the `generate_responses` method primarily returns a model identifier, and the actual responses are written to JSONL files in a specified directory. These files contain entries with the following structure:

```json
{
    "question_id": "question_identifier",
    "answer_id": "unique_answer_id",
    "model_id": "model_identifier",
    "choices": [
        {
            "index": 0,
            "turns": ["Turn 1 response", "Turn 2 response", ...] 
        }
    ],
    "tstamp": 1647894321.123
}
```

The evaluation happens using these saved responses rather than directly using the return value of `generate_responses`.

### Coding Benchmarks (HumanEval, MBPP, etc.)

Coding benchmarks typically return results in a format like:

```python
{
    "responses": ["Code solution 1", "Code solution 2", ...],
    "prompt_ids": ["prompt_id_1", "prompt_id_2", ...],
    "metadata": {...}  # Additional benchmark-specific metadata
}
```

### Reasoning/Math Benchmarks

These benchmarks often return results with detailed step-by-step reasoning:

```python
{
    "question_ids": ["q1", "q2", ...],
    "responses": ["Reasoning and answer for q1", "Reasoning and answer for q2", ...],
    "metadata": {...}
}
```

## Handling Variations in Post-Processing

When implementing post-processing logic that needs to work across all benchmarks, consider:

1. Check for common patterns first (`responses`, `choices`, etc.)
2. Add benchmark-specific handling for known formats (like MTBench)
3. Include fallback logic for unrecognized formats
4. Log warnings when encountering unexpected structures

The abstract structure suggests the actual post-processing implementation would benefit from a strategy pattern, with specific handlers for each major response format type.

## Note on Task Abstraction

The variation in response formats appears to be abstracted away in the benchmark classes. The main `evaluate()` function in `eval.py` handles these different formats through polymorphism - each benchmark class has its own implementation of `generate_responses()` and `evaluate_responses()` methods which hide the format details from the main evaluation pipeline.

This design allows for flexibility but requires careful handling when adding cross-benchmark functionality like reasoning post-processing.