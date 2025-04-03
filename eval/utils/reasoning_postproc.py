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


def extract_thinking_content(text: str, logger: logging.Logger) -> (str, bool, str):
    """
    Extract the content inside thinking tokens, if any.
    
    Args:
        text (str): The text to process
        logger (logging.Logger): Logger for tracking progress and errors
        
    Returns:
        tuple: (content inside thinking tokens, whether tokens were found, original text)
    """
    import re
    found_thinking = False
    
    # Define all thinking token patterns
    thinking_patterns = [
        # HTML-style tags
        (r'<think>(.*?)</think>', '<think>'),
        (r'<thinking>(.*?)</thinking>', '<thinking>'),
        (r'<thoughts>(.*?)</thoughts>', '<thoughts>'),
        (r'<thought>(.*?)</thought>', '<thought>'),
        (r'<Think>(.*?)</Think>', '<Think>'),
        (r'<Thinking>(.*?)</Thinking>', '<Thinking>'),
        (r'<Thoughts>(.*?)</Thoughts>', '<Thoughts>'),
        (r'<Thought>(.*?)</Thought>', '<Thought>'),
        
        # Special separator style tags
        (r'<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>', '<|begin_of_thought|>'),
        (r'<\|thinking\|>(.*?)<\|/thinking\|>', '<|thinking|>'),
        (r'<\|thought\|>(.*?)<\|/thought\|>', '<|thought|>'),
        (r'<\|thoughts\|>(.*?)<\|/thoughts\|>', '<|thoughts|>'),
        
        # Bracket style
        (r'\[thinking\](.*?)\[/thinking\]', '[thinking]'),
        (r'\[thought\](.*?)\[/thought\]', '[thought]'),
        (r'\[thoughts\](.*?)\[/thoughts\]', '[thoughts]'),
        (r'\[THINKING\](.*?)\[/THINKING\]', '[THINKING]'),
        (r'\[THOUGHT\](.*?)\[/THOUGHT\]', '[THOUGHT]'),
        (r'\[THOUGHTS\](.*?)\[/THOUGHTS\]', '[THOUGHTS]'),
    ]
    
    thinking_content = ""
    for pattern, tag_name in thinking_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            found_thinking = True
            logger.info(f"Found thinking content inside {tag_name} tags")
            
            # Concatenate all thinking content, preserving paragraph breaks
            for match in matches:
                if thinking_content:
                    thinking_content += "\n\n"  # Add paragraph break between different thinking blocks
                thinking_content += match.strip()
                
    if found_thinking:
        # Clean up any nested thinking tags - safer approach with explicit tags
        thinking_content = thinking_content.replace('<think>', '').replace('</think>', '')
        thinking_content = thinking_content.replace('<thinking>', '').replace('</thinking>', '')
        thinking_content = thinking_content.replace('<thoughts>', '').replace('</thoughts>', '')
        thinking_content = thinking_content.replace('<thought>', '').replace('</thought>', '')
        thinking_content = thinking_content.replace('<Think>', '').replace('</Think>', '')
        thinking_content = thinking_content.replace('<Thinking>', '').replace('</Thinking>', '')
        thinking_content = thinking_content.replace('<Thoughts>', '').replace('</Thoughts>', '')
        thinking_content = thinking_content.replace('<Thought>', '').replace('</Thought>', '')
        
        # Handle other format tags
        thinking_content = thinking_content.replace('<|begin_of_thought|>', '').replace('<|end_of_thought|>', '')
        thinking_content = thinking_content.replace('<|thinking|>', '').replace('<|/thinking|>', '')
        thinking_content = thinking_content.replace('<|thought|>', '').replace('<|/thought|>', '')
        thinking_content = thinking_content.replace('<|thoughts|>', '').replace('<|/thoughts|>', '')
        
        # Handle bracket formats
        thinking_content = thinking_content.replace('[thinking]', '').replace('[/thinking]', '')
        thinking_content = thinking_content.replace('[thought]', '').replace('[/thought]', '')
        thinking_content = thinking_content.replace('[thoughts]', '').replace('[/thoughts]', '')
        thinking_content = thinking_content.replace('[THINKING]', '').replace('[/THINKING]', '')
        thinking_content = thinking_content.replace('[THOUGHT]', '').replace('[/THOUGHT]', '')
        thinking_content = thinking_content.replace('[THOUGHTS]', '').replace('[/THOUGHTS]', '')
    
    return thinking_content, found_thinking, text


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
    
    # First extract any thinking content
    thinking_content, found_thinking, original_text = extract_thinking_content(text, logger)
    
    # If no thinking content was found, just do basic regex cleaning
    if not found_thinking:
        logger.info("No thinking tokens found, applying basic regex cleaning")
        return clean_thinking_tokens(text)
    
    # If thinking content was found, do regex cleaning to remove the tags
    cleaned_text = clean_thinking_tokens(text)
    logger.info("Using model to process thinking content even though regex cleaned the tags")
    
    # Check available GPU memory before trying model-based cleaning
    try:
        import torch
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            total_mem = torch.cuda.mem_get_info()[1] / (1024 ** 3)
            logger.info(f"Available GPU memory: {free_mem:.2f} GB free out of {total_mem:.2f} GB total")
            
            # If we have very little free VRAM, try one more round of cleanup
            if free_mem < 1.0:
                logger.warning(f"Only {free_mem:.2f} GB free VRAM available - attempting additional cleanup")
                
                # Force more aggressive garbage collection
                import gc
                for _ in range(3):  # Run multiple cycles
                    gc.collect()
                    torch.cuda.empty_cache()
                
                # Check if the cleanup helped
                free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                logger.info(f"After aggressive cleanup: {free_mem:.2f} GB free VRAM")
                
                # If still not enough memory, use regex-only cleaning
                if free_mem < 1.0:
                    logger.warning(f"Still only {free_mem:.2f} GB free VRAM - falling back to regex-only cleaning")
                    # Make sure we apply a very thorough regex cleaning as fallback
                    return clean_thinking_tokens(text)
    except Exception as e:
        logger.warning(f"Unable to check GPU memory: {str(e)}")
    
    # Process the thinking content with the model
    try:
        # Move model to GPU if it's currently on CPU
        model = ensure_model_on_gpu(model, logger)
            
        prompt = (
            "You are a helpful assistant that improves the quality of text. Your task is to refine this text "
            "to be more coherent, concise, and focused. Remove redundancies, unnecessary explanations, and "
            "self-referential language like 'wait', 'let me think', etc. Restructure the key points into a "
            "clear, helpful response. Here is the text to improve:\n\n"
            f"{thinking_content}\n\n"
            "Improved version:"
        )
        
        # Create messages in the format the model expects
        messages = [
            {"role": "system", "content": "You are a helpful assistant that improves the quality of text."},
            {"role": "user", "content": prompt}
        ]
        
        # Generate response with proper Instance object
        try:
            from lm_eval.api.instance import Instance
            
            # First get the templated input if possible
            try:
                inputs = model.apply_chat_template(messages)
            except Exception as template_err:
                logger.warning(f"Error applying chat template: {str(template_err)}")
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
            logger.info("Generating improved response using LLM for the thinking content")
            response = model.generate_until([instance])[0]
            
            if response and len(response) > 0:
                logger.info("Successfully generated improved response with LLM")
                return response.strip()
            else:
                logger.warning("Model produced empty response, returning regex-cleaned text")
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
    
    This function extracts the content inside thinking tokens if present,
    and uses a model to process this content. If no thinking tokens are found
    or model processing fails, it falls back to regex-based cleaning.
    
    Args:
        text (str): The text to post-process.
        postproc_model (Optional[LM]): Model to use for advanced post-processing.
        logger (Optional[logging.Logger]): Logger for tracking progress and errors.
        use_model (bool): Whether to use the model for advanced post-processing.
            
    Returns:
        str: Post-processed text with reasoning chains removed or improved.
    """
    if logger is None:
        logger = logging.getLogger("reasoning_postproc")
    
    # Skip processing if input is not a string
    if not isinstance(text, str):
        return text
    
    # Check if text contains thinking tokens
    thinking_content, has_thinking, _ = extract_thinking_content(text, logger)
    
    # If no thinking tokens were found, just return the original text
    if not has_thinking:
        logger.info("No thinking tokens found in text, returning as-is")
        return text
        
    # If thinking tokens were found but we don't have a model or don't want to use it,
    # fall back to regex cleaning
    if postproc_model is None or not use_model:
        logger.info("Thinking tokens found but no model available, using regex-only cleaning")
        return clean_thinking_tokens(text)
    
    # We found thinking tokens and have a model - process with the model
    # Regardless of whether regex would be effective, we want to use the model
    # to improve the content that was inside the thinking tags
    logger.info("Thinking tokens found, using model to process the content")
    
    try:
        # Clear CUDA cache before using model
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Clearing CUDA cache before model post-processing")
                torch.cuda.empty_cache()
        except (ImportError, Exception) as e:
            logger.warning(f"Unable to clear CUDA cache: {str(e)}")
            
        # Process with model
        result = process_with_model(postproc_model, text, logger)
        
        # If result contains no thinking tokens, we succeeded
        for pattern in ["<think>", "<thinking>", "<thought>", "[thinking]"]:
            if pattern in result:
                logger.warning(f"Model processing failed to remove {pattern} tags, applying regex cleanup")
                return clean_thinking_tokens(result)
                
        # Return model-processed result
        return result
    except Exception as e:
        logger.error(f"Error during model post-processing: {str(e)}")
        logger.warning("Falling back to regex-only cleaning")
        return clean_thinking_tokens(text)


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
        # Check if string contains thinking tokens - check for all common formats
        thinking_patterns = ["<think>", "<thinking>", "<thought>", "<Think>", 
                          "[thinking]", "[thought]", "<|thinking|>"]
        
        has_thinking = any(pattern in obj for pattern in thinking_patterns)
        if has_thinking:
            logger.info(f"Found thinking tokens in string, length: {len(obj)}")
        
        # Process the string
        result = postprocess_reasoning(obj, postproc_model, logger, use_model)
        
        # Check if processing was effective - check all patterns
        if has_thinking and not any(pattern in result for pattern in thinking_patterns):
            logger.info("Successfully removed thinking tokens from string")
        elif has_thinking and any(pattern in result for pattern in thinking_patterns):
            logger.warning("Failed to remove all thinking tokens from string")
            # Apply regex cleanup as a fallback
            result = clean_thinking_tokens(result)
            
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