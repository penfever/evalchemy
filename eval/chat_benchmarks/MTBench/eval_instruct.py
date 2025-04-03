from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import os
import random
import json
import time
import logging
import numpy as np
import pandas as pd
import shortuuid
import torch.distributed as dist
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.task import BaseBenchmark
from eval.eval import move_model_to_device, initialize_model
from eval.constants import THINK_PATTERNS as patterns
from eval.distributed.launch import cleanup_model
from fastchat.llm_judge.common import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    check_data,
    play_a_match_pair,
    play_a_match_single,
    NEED_REF_CATS,
    temperature_config,
)
from fastchat.utils import str_to_torch_dtype
from fastchat.llm_judge.gen_judgment import (
    make_match,
    make_match_all_pairs,
    make_match_single,
    make_judge_single,
    make_judge_pairwise,
)


@dataclass
class MTBenchConfig:
    """Configuration for MTBench evaluation."""

    bench_name: str = "mt_bench"
    question_begin: Optional[int] = None
    question_end: Optional[int] = None
    max_new_token: int = 1024
    num_choices: int = 1
    num_gpus_per_model: int = 1
    num_gpus_total: int = 1
    max_gpu_memory: Optional[str] = None
    dtype: Optional[str] = None
    revision: str = "main"
    judge_file: str = "eval/chat_benchmarks/MTBench/fastchat/llm_judge/data/judge_prompts.jsonl"
    judge_model: str = "gpt-4o-mini-2024-07-18"
    baseline_model: str = "gpt-3.5-turbo"
    mode: str = "single"
    parallel: int = 4
    first_n: Optional[int] = None


class MTBenchBenchmark(BaseBenchmark):
    """
    MTBench benchmark for evaluating multi-turn chat capabilities.
    """

    REQUIRES_OPENAI_ANNOTATOR = False  # Can also be anthropic

    def __init__(
        self,
        base_path: str = "eval/chat_benchmarks/MTBench",
        config: Optional[MTBenchConfig] = None,
        debug: bool = False,
        annotator_model: str = "gpt-4o-mini-2024-07-18",
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize MTBench benchmark.

        Args:
            base_path: Base directory for MTBench data and outputs
            config: MTBench configuration object
            debug: If True, run in debug mode on 2 samples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.base_path = Path(base_path)
        if annotator_model == "auto":
            annotator_model = "gpt-4"
        if config:
            print(f"Warning: Overwriting config.judge_model = {annotator_model} ")
            config.judge_model = annotator_model
        self.config = config or MTBenchConfig(judge_model=annotator_model)
        self.debug = debug

        # Setup paths
        self.data_path = self.base_path / "fastchat/llm_judge/data/mt_bench"
        self.question_file = self.data_path / "question.jsonl"
        self.answer_dir = self.data_path / "model_answer"
        self.ref_answer_dir = self.data_path / "reference_answer"
        self.judgment_dir = self.data_path / "model_judgment"

        # Create directories
        self.answer_dir.mkdir(parents=True, exist_ok=True)
        self.judgment_dir.mkdir(parents=True, exist_ok=True)

    def get_model_answers(self, model: LM, model_id: str, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate model answers for all questions."""
        # Initialize tracking structures
        all_convs = [[] for _ in questions]
        all_choices = [{"index": 0, "turns": []} for _ in questions]

        max_turns = max(len(q["turns"]) for q in questions)
        answer_file = self.answer_dir / f"{model_id}.jsonl"

        # Process each turn
        for turn_num in range(max_turns):
            self.logger.info(f"Processing Turn {turn_num + 1}")
            batch_instances = []

            # Prepare instances for current turn
            self.logger.info("Generating responses for MTBench...")
            for q_idx, question in enumerate(questions):
                if turn_num < len(question["turns"]):
                    temperature = temperature_config.get(question["category"], 0.7)

                    # Add user message to conversation
                    all_convs[q_idx].append({"role": "user", "content": question["turns"][turn_num]})

                    # Prepare model input
                    prompt = self._prepare_messages(all_convs[q_idx], model)
                    batch_instances.append(
                        Instance(
                            "generate_until",
                            all_convs[q_idx],
                            (
                                prompt,
                                {
                                    "max_gen_toks": self.config.max_new_token,
                                    "do_sample": temperature >= 1e-4,
                                    "temperature": temperature,
                                },
                            ),
                            q_idx,
                        )
                    )

            # Generate responses
            if batch_instances:
                outputs = self.compute(model, batch_instances)

                # Process outputs
                for q_idx, output in enumerate(outputs):
                    all_convs[q_idx].append({"role": "assistant", "content": output})
                    all_choices[q_idx]["turns"].append(output)

            if model.rank != 0:
                continue

            # Save completed conversations
            for q_idx, question in enumerate(questions):
                if turn_num == len(question["turns"]) - 1:
                    ans_json = {
                        "question_id": question["question_id"],
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_id,
                        "choices": [all_choices[q_idx]],
                        "tstamp": time.time(),
                    }
                    with open(answer_file, "a") as f:
                        f.write(json.dumps(ans_json) + "\n")

        return all_choices

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate responses for MTBench questions.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing model identifier, or None for non-primary ranks
        """
        # Load questions
        questions = load_questions(self.question_file, self.config.question_begin, self.config.question_end)

        if self.debug:
            questions = questions[:2]
            self.logger.info("Debug mode: using first 2 questions")

        # Shuffle questions for better load balancing
        random.shuffle(questions)

        # Generate answers
        answers = self.get_model_answers(model=model, model_id=model.model_identifier, questions=questions)

        # Return None early for non-primary ranks if compute() returned None
        if answers is None:
            return None

        return {"model_id": model.model_identifier, "answers": answers, "questions": questions}
        
    def generate_and_postprocess_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate responses for MTBench questions and postprocess thinking tokens.
        
        This method:
        1. Calls generate_responses to get model answers
        2. Detects thinking tokens (e.g., <thinking>...</thinking>) in the responses
        3. Uses a separate model to clean up the thinking blocks
        4. Returns the processed results
        
        Args:
            model: Language model instance
            
        Returns:
            Dictionary containing model identifier and processed answers, or None for non-primary ranks
        """
        # Store original model initialization parameters
        # Get model type (registry name, e.g., 'hf' or 'vllm')
        original_model_type = getattr(model.model, '_model_type', 'hf')
        
        # Get model arguments - ensure it's a string
        model_args = getattr(model, 'model_args', '')
        if model_args is None:
            model_args = ''
        elif not isinstance(model_args, str):
            model_args = str(model_args)
        
        # Extract model config to reuse initialization strategy
        model_batch_size = getattr(model, 'batch_size', None)
        if not model_batch_size:
            model_batch_size = getattr(model, 'batch_size_per_gpu', None)
            
        # Call the regular generate_responses method
        result = self.generate_responses(model)
        
        # Return None early for non-primary ranks if generate_responses returned None
        if result is None:
            return None
            
        # Extract answers and questions
        model_id = result["model_id"]
        answers = result["answers"]
        questions = result["questions"]
        
        # Check if any response contains thinking tokens
        needs_postprocessing = False
        for choice in answers:
            for turn_response in choice["turns"]:
                for pattern in patterns:
                    import re 
                    if re.search(pattern, turn_response, re.DOTALL):
                        needs_postprocessing = True
                        self.logger.info(f"Found thinking tokens in response, will apply postprocessing")
                        break
                if needs_postprocessing:
                    break
            if needs_postprocessing:
                break
                
        # If we're in debug mode, force postprocessing
        if self.debug and not needs_postprocessing:
            needs_postprocessing = True
            self.logger.info("Debug mode: Forcing postprocessing even without thinking tokens")
            
        # If we need postprocessing, initialize the postprocessing model
        if needs_postprocessing:
            
            # Clean up the model manually rather than trying to move it
            self.logger.info(f"Cleaning up main model ({original_model_type})")
            print("Model methods: ", dir(model.model))
            print("Model type: ", type(model.model))
            
            # Try a drastic approach: explicitly release all VLLM components
            if hasattr(model, 'model') and hasattr(model.model, 'llm_engine'):
                self.logger.info("Attempting forceful VLLM memory cleanup")
                vllm_obj = model.model
                try:
                    # Try to access block manager to release memory blocks
                    if hasattr(vllm_obj.llm_engine, 'block_manager'):
                        self.logger.info("Attempting to free VLLM block manager memory")
                        if hasattr(vllm_obj.llm_engine.block_manager, 'free_all_blocks'):
                            self.logger.info("Calling free_all_blocks")
                            vllm_obj.llm_engine.block_manager.free_all_blocks()
                        
                    # Try to explicitly destroy cache managers
                    if hasattr(vllm_obj.llm_engine, 'cache_manager'):
                        self.logger.info("Resetting cache manager")
                        vllm_obj.llm_engine.cache_manager = None
                    
                    # Try to wake up and then force termination
                    if hasattr(vllm_obj, 'wake_up') and callable(vllm_obj.wake_up):
                        self.logger.info("Waking up engine to reset state")
                        vllm_obj.wake_up()
                    
                    # Force clean cache if possible
                    if hasattr(vllm_obj, 'reset_prefix_cache') and callable(vllm_obj.reset_prefix_cache):
                        self.logger.info("Forcing prefix cache reset multiple times")
                        for _ in range(3):
                            vllm_obj.reset_prefix_cache()
                except Exception as e:
                    self.logger.warning(f"Error in forceful VLLM memory cleanup: {e}")
            
            # Try the regular cleanup
            cleanup_model(model)
            
            # Try the nuclear option: force all cuda caches to reset
            try:
                import torch
                if torch.cuda.is_available():
                    import os
                    # Set environment variables to restrict memory usage for next model
                    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
                    
                    # Force CUDA synchronization and reset
                    torch.cuda.synchronize()
                    for device in range(torch.cuda.device_count()):
                        torch.cuda.set_device(device)
                        torch.cuda.empty_cache()
                    
                    # On some systems we can explicitly release device memory
                    if hasattr(torch.cuda, 'memory_deallocate'):
                        self.logger.info("Explicitly deallocating all CUDA memory")
                        torch.cuda.memory_deallocate()
            except Exception as e:
                self.logger.warning(f"Error in CUDA memory reset: {e}")
            
            # Last resort: try to force a GPU reset by briefly switching to CPU
            # This will at least detect if there's a memory leak
            try:
                import torch
                if torch.cuda.is_available():
                    # Create a small dummy tensor on CPU then GPU to test memory
                    self.logger.info("Creating test tensor to verify GPU memory availability")
                    test_tensor = torch.zeros((1, 1), device="cpu")
                    try:
                        # Try to move to GPU - this will fail if we're truly out of memory
                        test_tensor = test_tensor.cuda()
                        self.logger.info("Successfully created test tensor on GPU")
                        del test_tensor
                    except RuntimeError as e:
                        self.logger.error(f"Failed to allocate test tensor: {e}")
                        self.logger.error("GPU memory is not being properly freed!")
            except Exception as e:
                self.logger.warning(f"Error testing GPU memory: {e}")
                
            # Initialize the postprocessing model using the same strategy as the original model
            self.logger.info(f"Initializing postprocessing model: {self.reasoning_postproc_model}")
            
            # Reuse the same initialization strategy and parameters from the original model
            postproc_args = f"pretrained={self.reasoning_postproc_model}"
            
            # Extract parameters from model args safely
            import re
            
            # Ensure model_args is a string for regex operations
            model_args_str = str(model_args) if model_args is not None else ""
            
            # Extract dtype
            try:
                dtype_match = re.search(r'dtype=([^,]+)', model_args_str)
                if dtype_match:
                    postproc_args += f",dtype={dtype_match.group(1)}"
                else:
                    # Default to bfloat16 if not specified
                    postproc_args += ",dtype=bfloat16"
            except Exception as e:
                self.logger.warning(f"Error extracting dtype: {e}")
                postproc_args += ",dtype=bfloat16"
                
            # Add batch size if available
            if model_batch_size:
                postproc_args += f",batch_size={model_batch_size}"
                
            # Add other important parameters from original model (except pretrained)
            for param in ["tp_size", "parallelize", "max_memory_per_gpu"]:
                try:
                    param_match = re.search(f'{param}=([^,]+)', model_args_str)
                    if param_match:
                        postproc_args += f",{param}={param_match.group(1)}"
                except Exception as e:
                    self.logger.warning(f"Error extracting {param}: {e}")
            
            self.logger.info(f"Initializing postprocessing model with args: {postproc_args}")
            try:
                # Always use 'hf' for the postprocessing model, regardless of original model type
                postproc_model = initialize_model(
                    model="hf",  # Always use HF for postprocessing
                    model_args=postproc_args,
                    device="cuda"
                )
            except Exception as e:
                self.logger.error(f"Error initializing postprocessing model: {e}")
                self.logger.info("Trying with simplified parameters...")
                # Fallback with minimal parameters
                postproc_model = initialize_model(
                    model="hf",
                    model_args=f"pretrained={self.reasoning_postproc_model},dtype=float32",
                    device="cuda"
                )
            
            # Process each response
            self.logger.info("Postprocessing responses with thinking tokens")
            
            for choice_idx, choice in enumerate(answers):
                for turn_idx, turn_response in enumerate(choice["turns"]):
                    processed_response = turn_response
                    
                    # Look for thinking tokens using each pattern
                    for pattern in patterns:
                        matches = list(re.finditer(pattern, processed_response, re.DOTALL))
                        
                        # Process from end to beginning to avoid messing up indices
                        for match in reversed(matches):
                            start, end = match.span()
                            thinking_block = processed_response[start:end]
                            
                            # Skip very short thinking blocks (likely false positives)
                            if len(thinking_block) < 20:
                                continue
                                
                            # Prepare prompt for the postprocessing model
                            prompt_messages = [
                                {"role": "system", "content": "You are a helpful AI assistant that helps clean up thinking processes in text. Remove all special tokens and clean up the text to make it concise, coherent, and well-structured."},
                                {"role": "user", "content": f"This is a thinking block from an AI assistant's response. Please clean it up by:\n1. Removing all thinking tokens and markers\n2. Removing repetitive, uncertain, or rambling text\n3. Making it concise and clear\n4. Preserving the core insights\n\nHere's the thinking block:\n\n{thinking_block}"}
                            ]
                            
                            # Apply chat template (if model supports it)
                            prompt = None
                            if hasattr(postproc_model, "apply_chat_template") and callable(postproc_model.apply_chat_template):
                                prompt = postproc_model.apply_chat_template(prompt_messages)
                            else:
                                # Basic fallback if model doesn't support chat templates
                                prompt = prompt_messages[-1]["content"]
                            
                            # Create instance
                            instance = Instance(
                                "generate_until",
                                prompt_messages,
                                (
                                    prompt,
                                    {
                                        "max_gen_toks": 1024,
                                        "do_sample": False,
                                        "temperature": 0.0,
                                    },
                                ),
                                0,
                            )
                            
                            # Generate cleaned response
                            cleaned_thinking = self.compute(postproc_model, [instance])[0]
                            
                            # Replace the thinking block with cleaned output
                            processed_response = processed_response[:start] + cleaned_thinking + processed_response[end:]
                    
                    # Update the answer with the processed response
                    choice["turns"][turn_idx] = processed_response
            
            # Save the processed answers to a different file for comparison
            if model.rank == 0:
                answer_file = self.answer_dir / f"{model_id}_processed.jsonl"
                import time
                with open(answer_file, "w") as f:
                    for q_idx, question in enumerate(questions):
                        if q_idx < len(answers):
                            ans_json = {
                                "question_id": question["question_id"],
                                "answer_id": shortuuid.uuid(),
                                "model_id": model_id,
                                "choices": [answers[q_idx]],
                                "tstamp": time.time(),
                            }
                            f.write(json.dumps(ans_json) + "\n")
            
            # Clean up the postprocessing model to free up GPU memory
            self.logger.info("Cleaning up postprocessing model")
            
            # For VLLM models, try to shutdown the engine if possible
            if hasattr(postproc_model, 'engine') and hasattr(postproc_model.engine, 'shutdown'):
                try:
                    self.logger.info("Detected VLLM postprocessing model, shutting down engine")
                    postproc_model.engine.shutdown()
                except Exception as e:
                    self.logger.warning(f"Failed to shutdown VLLM engine: {e}")
                    
            # Try to call any available cleanup methods
            for cleanup_method in ['close', 'cleanup', 'shutdown', 'terminate']:
                if hasattr(postproc_model, cleanup_method) and callable(getattr(postproc_model, cleanup_method)):
                    try:
                        self.logger.info(f"Calling {cleanup_method}() method on postprocessing model")
                        getattr(postproc_model, cleanup_method)()
                    except Exception as e:
                        self.logger.warning(f"Error calling {cleanup_method}(): {e}")
            
            # Delete the postprocessing model
            del postproc_model
            
            # More aggressive memory cleanup
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    # Empty cache multiple times
                    for _ in range(3):
                        torch.cuda.empty_cache()
                    
                    # Try to reset peak memory stats
                    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                        torch.cuda.reset_peak_memory_stats()
                        
                    # Try a more aggressive approach
                    if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                        torch.cuda.reset_accumulated_memory_stats()
            except (ImportError, AttributeError) as e:
                self.logger.warning(f"Error clearing CUDA memory: {e}")
                
            # Try to sleep a bit to allow OS to reclaim memory
            import time
            time.sleep(1)
            
            # Recreate the main model with original parameters
            self.logger.info(f"Recreating main model of type {original_model_type} with original parameters")
            self.logger.info(f"Model args: {model_args}")
            
            # Initialize with original parameters - ensure model_args is a string
            model_args_str = str(model_args) if model_args is not None else ""
            model = initialize_model(
                model=original_model_type,
                model_args=model_args_str,
                device="cuda",
                batch_size=model_batch_size
            )
            
        # Return only the model_id for compatibility with evaluate_responses
        return {"model_id": model_id}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate model responses using GPT-4 judge.

        Args:
            results: Dictionary containing model identifier and optionally answers and questions

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        # Handle None result from non-primary ranks
        if results is None:
            return None

        # Check if we should use the processed answer file
        model_id = results["model_id"]
        processed_answer_file = self.answer_dir / f"{model_id}_processed.jsonl"
        use_processed_answers = processed_answer_file.exists()
        
        if use_processed_answers:
            self.logger.info(f"Found processed answers file at {processed_answer_file}, will use for evaluation")

        # Load data
        questions = load_questions(self.question_file, None, None)
        if self.debug:
            questions = questions[:2]
            self.logger.info(f"Debug mode: using 2 examples")

        # Load answers, using processed answers if available
        model_answers = load_model_answers(self.answer_dir)
        if use_processed_answers:
            # Add processed answers to the model_answers dictionary
            processed_answers = {}
            with open(processed_answer_file, "r") as f:
                for line in f:
                    answer = json.loads(line)
                    question_id = answer["question_id"]
                    processed_answers[question_id] = answer
            
            # Replace the original model answers with processed answers
            for question_id, answer in processed_answers.items():
                if question_id in model_answers[model_id]:
                    model_answers[model_id][question_id] = answer
            
        ref_answers = load_model_answers(self.ref_answer_dir)
        judge_prompts = load_judge_prompts(self.config.judge_file)

        # Setup evaluation
        models = [results["model_id"]]
        if self.config.mode == "single":
            judges = make_judge_single(self.config.judge_model, judge_prompts)
            play_a_match_func = play_a_match_single
            output_file = self.judgment_dir / "gpt-4_single.jsonl"
            make_match_func = make_match_single
            baseline_model = None
        else:
            judges = make_judge_pairwise(self.config.judge_model, judge_prompts)
            play_a_match_func = play_a_match_pair
            output_file = self.judgment_dir / "gpt-4_pair.jsonl"
            if self.config.mode == "pairwise-all":
                make_match_func = make_match_all_pairs
                baseline_model = None
            else:
                make_match_func = make_match
                baseline_model = self.config.baseline_model

        # Verify data
        check_data(questions, model_answers, ref_answers, models, judges)

        # Split questions by category
        questions_math = [q for q in questions if q["category"] in NEED_REF_CATS]
        questions_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

        # Create matches
        matches = []
        matches.extend(make_match_func(questions_default, models, model_answers, judges["default"], baseline_model))
        matches.extend(
            make_match_func(
                questions_math,
                models,
                model_answers,
                judges["math"],
                baseline_model,
                ref_answers,
            )
        )
        matches.extend(
            make_match_func(
                questions_default,
                models,
                model_answers,
                judges["default-mt"],
                baseline_model,
                multi_turn=True,
            )
        )
        matches.extend(
            make_match_func(
                questions_math,
                models,
                model_answers,
                judges["math-mt"],
                baseline_model,
                ref_answers,
                multi_turn=True,
            )
        )

        # Run evaluation
        if self.config.parallel == 1:
            for match in tqdm(matches):
                play_a_match_func(match, output_file=output_file)
        else:
            with ThreadPoolExecutor(self.config.parallel) as executor:
                list(
                    tqdm(
                        executor.map(lambda m: play_a_match_func(m, output_file=output_file), matches),
                        total=len(matches),
                    )
                )

        # Load and process results
        df_all = pd.read_json(output_file, lines=True)
        df = df_all[["model", "score", "turn"]]
        df = df[df["score"] != -1]

        # Calculate scores
        df_turn1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
        df_turn2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
        df_avg = df[["model", "score"]].groupby(["model"]).mean()

        model_id = results["model_id"]
        return {
            "Turn 1": df_turn1.loc[results["model_id"]].score.values[0],
            "Turn 2": df_turn2.loc[results["model_id"]].score.values[0],
            "Average": df_avg.loc[results["model_id"]].score,
        }

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete MTBench evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        self.logger.info("Starting MTBench evaluation")
        try:
            generation_results = self.generate_responses(model)

            # If not primary rank, return None early
            if generation_results is None:
                return None

            evaluation_results = self.evaluate_responses(generation_results)
            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
