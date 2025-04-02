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
    if hasattr(model, '_is_on_cpu') and model._is_on_cpu:
        if logger:
            logger.info("Moving post-processing model from CPU to GPU for inference")
        
        # Different model types need different approaches to move to GPU
        if hasattr(model, 'model') and hasattr(model.model, 'to'):
            model.model = model.model.to('cuda')
            
        elif hasattr(model, 'model') and hasattr(model.model, 'cuda'):
            model.model.cuda()
            
        # Also move tokenizer to GPU if possible
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'to'):
            try:
                model.tokenizer = model.tokenizer.to('cuda')
            except:
                pass  # Ignore if tokenizer can't be moved
                
        # Mark as now on GPU
        model._is_on_cpu = False
        
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
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',
        r'<thoughts>.*?</thoughts>',
        r'<\|begin_of_thought\|>.*?<\|end_of_thought\|>',
        r'<\|thinking\|>.*?<\|/thinking\|>',
        r'\[thinking\].*?\[/thinking\]',
        r'<thinking>\n?.*?\n?</thinking>',
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
    
    try:
        # Create messages in the format the model expects
        messages = [
            {"role": "system", "content": "You are a helpful assistant that cleans up text."},
            {"role": "user", "content": prompt}
        ]
        
        # Generate response
        inputs = model.apply_chat_template(messages)
        response = model.generate_until([inputs])[0]
        
        if response and len(response) > 0:
            return response.strip()
        else:
            logger.warning("Model produced empty response, returning original text")
            return text
    except Exception as e:
        logger.error(f"Error processing with model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return text


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
    
    # If model is provided and use_model is True, also use model for more advanced processing
    if postproc_model is not None and use_model:
        try:
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
    
    # Process strings
    if isinstance(obj, str):
        return postprocess_reasoning(obj, postproc_model, logger, use_model)
    
    # Process lists
    elif isinstance(obj, list):
        return [postprocess_object(item, postproc_model, logger, use_model) for item in obj]
    
    # Process dictionaries
    elif isinstance(obj, dict):
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
        return obj