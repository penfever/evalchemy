"""
Reasoning post-processing utilities for cleaning model responses.

This module provides functions to clean up model responses by removing
thinking tokens, self-referential language, and repetitive reasoning chains.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union

from lm_eval.api.model import LM


def initialize_reasoning_postprocessor(model_name: str, use_cpu: bool = True, **model_kwargs) -> LM:
    """
    Initialize a model for reasoning post-processing.
    
    Args:
        model_name (str): Name of the model to use for post-processing.
        use_cpu (bool): Whether to load the model on CPU initially to save GPU memory.
            If True, the model is loaded to CPU and will be moved to GPU when needed.
        **model_kwargs: Additional arguments to pass to the model initialization.
            
    Returns:
        LM: Initialized language model instance for post-processing.
    """
    from lm_eval.api.registry import get_model
    from lm_eval.utils import sanitize_model_name
    
    # Default to bfloat16 if not specified
    if "dtype" not in model_kwargs:
        model_kwargs["dtype"] = "bfloat16"
    
    # Default to batch size 1 if not specified
    if "batch_size" not in model_kwargs:
        model_kwargs["batch_size"] = 1
        
    # Create model args string
    model_args = f"pretrained={model_name}"
    for key, value in model_kwargs.items():
        model_args += f",{key}={value}"
    
    # Load to CPU initially to avoid VRAM usage until needed
    device = "cpu" if use_cpu else "cuda"
    config = {"device": device}
    
    # Set low_cpu_mem_usage to True for more efficient CPU loading
    if "low_cpu_mem_usage" not in model_kwargs and use_cpu:
        model_args += ",low_cpu_mem_usage=True"
        
    lm = get_model("hf").create_from_arg_string(model_args, config)
    lm.model_identifier = sanitize_model_name(f"model_hf_model_args_{model_args}")
    
    # Store whether the model is currently on CPU
    lm._is_on_cpu = use_cpu
    
    return lm

def ensure_model_on_gpu(model: LM, logger: Optional[logging.Logger] = None) -> LM:
    """
    Ensure the model is moved to GPU before using it.
    
    Args:
        model: The model to ensure is on GPU
        logger: Optional logger for debugging
        
    Returns:
        The model, now on GPU if it wasn't already
    """
    if not hasattr(model, '_is_on_cpu') or not model._is_on_cpu:
        # Model is already on GPU or doesn't track CPU/GPU state
        return model
        
    if logger:
        logger.info("Moving post-processing model from CPU to GPU for inference")
    
    # Check available GPU memory first
    try:
        import torch
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            total_mem = torch.cuda.mem_get_info()[1] / (1024 ** 3)
            if logger:
                logger.info(f"Before moving model to GPU: {free_mem:.2f} GB free out of {total_mem:.2f} GB total VRAM")
            
            # If we have very little VRAM available, don't even try to move the model
            if free_mem < 1.0:
                if logger:
                    logger.warning(f"Less than 1 GB free VRAM available ({free_mem:.2f} GB) - keeping model on CPU")
                return model
                
            # Try to free CUDA cache to reduce OOM risk
            torch.cuda.empty_cache()
            if logger:
                logger.info("Cleared CUDA cache before moving model to GPU")
    except (ImportError, Exception) as e:
        if logger:
            logger.warning(f"Unable to check/clear GPU memory: {str(e)}")
    
    # Try to move model components to GPU
    try:
        # Different model types need different approaches to move to GPU
        if hasattr(model, 'model'):
            if hasattr(model.model, 'to'):
                model.model = model.model.to('cuda')
                if logger:
                    logger.info("Moved model to GPU using .to('cuda')")
            elif hasattr(model.model, 'cuda'):
                model.model.cuda()
                if logger:
                    logger.info("Moved model to GPU using .cuda()")
            
        # Also move tokenizer to GPU if possible
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'to'):
            try:
                model.tokenizer = model.tokenizer.to('cuda')
                if logger:
                    logger.info("Moved tokenizer to GPU")
            except Exception as tokenizer_e:
                if logger:
                    logger.debug(f"Could not move tokenizer to GPU: {str(tokenizer_e)}")
                
        # Mark as now on GPU
        model._is_on_cpu = False
        if logger:
            logger.info("Successfully moved post-processing model to GPU")
            
        # Check available memory after moving
        try:
            import torch
            if torch.cuda.is_available():
                free_mem_after = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                if logger:
                    logger.info(f"After moving model to GPU: {free_mem_after:.2f} GB free VRAM")
        except Exception:
            pass
            
    except Exception as e:
        if logger:
            logger.error(f"Failed to move model to GPU: {str(e)}")
            logger.warning("Will continue with model on CPU")
            import traceback
            logger.error(traceback.format_exc())
    
    return model


def clean_thinking_tokens(text: str) -> str:
    """
    Clean thinking tokens from model output using regex.
    
    Args:
        text (str): The model output text to clean.
        
    Returns:
        str: Cleaned text with thinking tokens removed.
    """
    # Skip cleaning if input is not a string
    if not isinstance(text, str):
        return text
        
    # Remove common thought markers
    patterns = [
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
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    return text.strip()


def process_with_model(model: LM, text: str, logger: logging.Logger) -> str:
    """
    Process text with a model to remove reasoning chains and self-referential language.
    
    Args:
        model (LM): The model to use for processing.
        text (str): The text to process.
        logger (logging.Logger): Logger for tracking progress and errors.
        
    Returns:
        str: Processed text with reasoning chains and self-references removed.
    """
    # Skip processing if input is not a string
    if not isinstance(text, str):
        return text
    
    # Try using regex-based cleaning first to reduce reliance on model
    cleaned_text = clean_thinking_tokens(text)
    if cleaned_text != text:
        logger.info("Successfully removed thinking tokens with regex, skipping model-based cleaning")
        return cleaned_text
    
    # Check available GPU memory before trying model-based cleaning
    try:
        import torch
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            total_mem = torch.cuda.mem_get_info()[1] / (1024 ** 3)
            logger.info(f"Available GPU memory: {free_mem:.2f} GB free out of {total_mem:.2f} GB total")
            
            # If we have very little free VRAM, skip model-based cleaning
            if free_mem < 1.0:
                logger.warning(f"Only {free_mem:.2f} GB free VRAM available - skipping model-based cleaning")
                return cleaned_text
    except Exception as e:
        logger.warning(f"Unable to check GPU memory: {str(e)}")
    
    # If regex didn't help, try model-based cleaning
    try:
        # Move model to GPU if it's currently on CPU
        model = ensure_model_on_gpu(model, logger)
            
        prompt = (
            "You are a helpful assistant that cleans up text to remove internal reasoning chains and "
            "self-referential language. Your task is to copy the entire response, but remove all internal "
            "reasoning chains that are repetitive or that did not lead to the ultimate answer. Also remove "
            "self-referential talk like 'wait', 'stop', 'I'm not sure', 'let's think', etc. Keep the final "
            "answer and any essential context. Here is the text to clean up:\n\n"
            f"{text}\n\n"
            "Cleaned version (keep line breaks and formatting, just remove reasoning chains and self-references):"
        )
        
        # Create messages in the format the model expects
        messages = [
            {"role": "system", "content": "You are a helpful assistant that cleans up text."},
            {"role": "user", "content": prompt}
        ]
        
        # Generate response with proper Instance object
        try:
            from lm_eval.api.instance import Instance
            
            # First get the templated input if possible
            try:
                inputs = model.apply_chat_template(messages)
            except:
                # If apply_chat_template fails, use the prompt directly
                inputs = prompt
                
            # Create a proper Instance object with the required args format
            instance = Instance(
                task_name="generate_until",
                messages=messages,
                args=(inputs, {"max_new_tokens": 1024}),
                idx=0
            )
            
            # Generate response
            logger.info("Generating response using LLM for post-processing")
            response = model.generate_until([instance])[0]
            
            if response and len(response) > 0:
                logger.info("Successfully generated post-processed response with LLM")
                return response.strip()
            else:
                logger.warning("Model produced empty response, returning original text")
                return cleaned_text
                
        except Exception as e:
            logger.warning(f"Error using generate_until with Instance: {str(e)}")
            logger.warning("Falling back to regex-only cleaning")
            return cleaned_text
            
    except Exception as e:
        logger.error(f"Error during model-based post-processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return cleaned_text


def postprocess_reasoning(
    text: str, 
    postproc_model: Optional[LM] = None, 
    logger: Optional[logging.Logger] = None,
    use_model: bool = True
) -> str:
    """
    Post-process model response to clean up reasoning chains.
    
    This function first applies regex patterns to remove thinking tokens,
    then optionally uses a model to further clean up reasoning chains and
    self-referential language.
    
    Args:
        text (str): The text to post-process.
        postproc_model (Optional[LM]): Model to use for advanced post-processing.
        logger (Optional[logging.Logger]): Logger for tracking progress and errors.
        use_model (bool): Whether to use the model for advanced post-processing.
            
    Returns:
        str: Post-processed text with reasoning chains removed.
    """
    if logger is None:
        logger = logging.getLogger("reasoning_postproc")
    
    # Skip processing if input is not a string
    if not isinstance(text, str):
        return text
    
    # First apply regex cleaning to remove thinking tokens
    cleaned_text = clean_thinking_tokens(text)
    
    # If the regex cleaning was effective, avoid using the model to save resources
    if cleaned_text != text and ("<think>" not in cleaned_text) and ("<thinking>" not in cleaned_text):
        logger.info("Regex cleaning was effective, skipping model-based post-processing")
        return cleaned_text
    
    # If model is provided and use_model is True, also use model for more advanced processing
    # But only if regex wasn't enough
    if postproc_model is not None and use_model:
        try:
            # Clear CUDA cache before using model
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("Clearing CUDA cache before model post-processing")
                    torch.cuda.empty_cache()
            except (ImportError, Exception) as e:
                logger.warning(f"Unable to clear CUDA cache: {str(e)}")
                
            return process_with_model(postproc_model, cleaned_text, logger)
        except Exception as e:
            logger.error(f"Error during model post-processing: {str(e)}")
            return cleaned_text
    
    return cleaned_text


def postprocess_object(
    obj: Any, 
    postproc_model: Optional[LM] = None, 
    logger: Optional[logging.Logger] = None,
    use_model: bool = True
) -> Any:
    """
    Post-process an object containing model responses.
    
    Recursively traverses dictionaries, lists, and strings to apply
    reasoning post-processing to all string values.
    
    Args:
        obj (Any): The object to post-process.
        postproc_model (Optional[LM]): Model to use for advanced post-processing.
        logger (Optional[logging.Logger]): Logger for tracking progress and errors.
        use_model (bool): Whether to use the model for advanced post-processing.
            
    Returns:
        Any: Post-processed object with reasoning chains removed from all strings.
    """
    if logger is None:
        logger = logging.getLogger("reasoning_postproc")
        
    logger.info(f"postprocess_object called on type: {type(obj).__name__}")
    
    # Process strings
    if isinstance(obj, str):
        # Check if string contains thinking tokens
        has_thinking = "<think>" in obj
        if has_thinking:
            logger.info(f"Found thinking tokens in string, length: {len(obj)}")
        
        # Process the string
        result = postprocess_reasoning(obj, postproc_model, logger, use_model)
        
        # Check if processing was effective
        if has_thinking and "<think>" not in result:
            logger.info("Successfully removed thinking tokens from string")
        elif has_thinking and "<think>" in result:
            logger.warning("Failed to remove thinking tokens from string")
            
        return result
    
    # Process lists
    elif isinstance(obj, list):
        logger.info(f"Processing list with {len(obj)} items")
        return [postprocess_object(item, postproc_model, logger, use_model) for item in obj]
    
    # Process dictionaries
    elif isinstance(obj, dict):
        logger.info(f"Processing dict with keys: {list(obj.keys())}")
        
        # Special handling for MTBench answer objects
        if "choices" in obj and isinstance(obj["choices"], list):
            logger.info("Found 'choices' key, this looks like an MTBench answer object")
            for choice in obj.get("choices", []):
                if "turns" in choice and isinstance(choice["turns"], list):
                    for i, turn in enumerate(choice["turns"]):
                        if isinstance(turn, str) and "<think>" in turn:
                            logger.info(f"Found thinking tokens in turn {i}")
        
        result = {}
        for key, value in obj.items():
            # Skip some metadata fields that shouldn't be modified
            if key in ["model_id", "question_id", "answer_id", "tstamp", "metadata"]:
                result[key] = value
            else:
                result[key] = postprocess_object(value, postproc_model, logger, use_model)
        return result
    
    # Return other types unchanged
    else:
        logger.info(f"Skipping object of type {type(obj).__name__}")
        return obj