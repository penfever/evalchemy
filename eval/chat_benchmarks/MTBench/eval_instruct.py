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
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.task import BaseBenchmark
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
        reasoning_postproc: bool = False,
        reasoning_postproc_model: str = "Qwen/Qwen2.5-7B-Instruct",
    ):
        """
        Initialize MTBench benchmark.

        Args:
            base_path: Base directory for MTBench data and outputs
            config: MTBench configuration object
            debug: If True, run in debug mode on 10 samples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
            reasoning_postproc: Whether to enable reasoning post-processing
            reasoning_postproc_model: Model to use for reasoning post-processing
        """
        super().__init__(
            logger=logger, 
            system_instruction=system_instruction,
            reasoning_postproc=reasoning_postproc,
            reasoning_postproc_model=reasoning_postproc_model
        )
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
        
        if self.reasoning_postproc:
            self.logger.info(f"MTBench reasoning post-processing is enabled with model: {self.reasoning_postproc_model}")

    def get_model_answers(self, model: LM, model_id: str, questions: List[Dict[str, Any]]) -> Optional[bool]:
        """
        Generate model answers for all questions and save directly to disk.
        
        Returns True on success, None on non-primary ranks, or False on error.
        """
        # Initialize tracking structures
        all_convs = [[] for _ in questions]
        all_choices = [{"index": 0, "turns": []} for _ in questions]

        max_turns = max(len(q["turns"]) for q in questions)
        answer_file = self.answer_dir / f"{model_id}.jsonl"
        
        # If primary rank and file exists, remove it to avoid duplicates
        if model.rank == 0 and answer_file.exists():
            answer_file.unlink()

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

            # Save completed conversations immediately to disk (for memory efficiency)
            for q_idx, question in enumerate(questions):
                if turn_num == len(question["turns"]) - 1:
                    ans_json = {
                        "question_id": question["question_id"],
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_id,
                        "choices": [all_choices[q_idx]],
                        "tstamp": time.time(),
                    }
                    # Write directly to file instead of storing in memory
                    with open(answer_file, "a") as f:
                        f.write(json.dumps(ans_json) + "\n")

        # Return None for non-primary ranks, True for success on primary rank
        return True if model.rank == 0 else None

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
            questions = questions[:10]
            self.logger.info("Debug mode: using first 10 questions")

        # Shuffle questions for better load balancing
        random.shuffle(questions)

        # Generate answers and write directly to disk
        # We minimize in-memory storage for VRAM efficiency
        _ = self.get_model_answers(model=model, model_id=model.model_identifier, questions=questions)

        # Return None early for non-primary ranks if compute() returned None
        if _ is None:
            return None

        # Only return minimal data needed for evaluation
        return {
            "model_id": model.model_identifier
        }

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate model responses using GPT-4 judge.

        Args:
            results: Dictionary containing model identifier

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        # Handle None result from non-primary ranks
        if results is None:
            return None

        # Check if we need to apply reasoning post-processing
        # This is important because the framework might be calling this method directly
        # instead of going through run_benchmark where post-processing normally happens
        if self.reasoning_postproc:
            self.logger.info("Post-processing is enabled. Checking if it needs to be applied...")
            try:
                # Initialize the post-processing model if needed
                if self._ensure_postproc_model_loaded():
                    # Process the answer file
                    model_id = results["model_id"]
                    answer_file = self.answer_dir / f"{model_id}.jsonl"
                    
                    if answer_file.exists():
                        # Check if file contains any variation of thinking tokens
                        needs_processing = False
                        
                        # Define all thinking token patterns to check for
                        thinking_patterns = [
                            "<think>", "<thinking>", "<thoughts>", "<thought>",
                            "<Think>", "<Thinking>", "<Thoughts>", "<Thought>",
                            "<|thinking|>", "<|thought|>", "<|thoughts|>",
                            "<|begin_of_thought|>", 
                            "[thinking]", "[thought]", "[thoughts]",
                            "[THINKING]", "[THOUGHT]", "[THOUGHTS]"
                        ]
                        
                        with open(answer_file, "r") as f:
                            # If in debug mode, force processing of the first line
                            if self.debug:
                                self.logger.info("Debug mode enabled, forcing post-processing of the first line")
                                needs_processing = True
                            else:
                                for line in f:
                                    if any(pattern in line for pattern in thinking_patterns):
                                        self.logger.info(f"Found thinking token pattern in file")
                                        needs_processing = True
                                        break
                        
                        if needs_processing:
                            self.logger.info(f"Applying post-processing to {answer_file}...")
                            
                            # Load answers from disk
                            with open(answer_file, "r") as f:
                                answers = [json.loads(line) for line in f]
                            
                            # Define the regex patterns for various thinking tokens
                            from eval.utils.reasoning_postproc import clean_thinking_tokens
                            
                            # Process answers
                            processed_answers = []
                            for ans in answers:
                                processed_ans = ans.copy()
                                has_thinking_tokens = False
                                
                                # Process each choice's turn content
                                if "choices" in processed_ans:
                                    for choice_idx, choice in enumerate(processed_ans["choices"]):
                                        if "turns" in choice:
                                            for turn_idx, turn_content in enumerate(choice["turns"]):
                                                # More robust check for thinking tokens - only match exact patterns
                                                contains_thinking = False
                                                for pattern in thinking_patterns:
                                                    if pattern in turn_content:
                                                        # Double-check that it's not a false positive by checking for closing tags
                                                        # For tags like <think>, check for </think>
                                                        if pattern.startswith("<") and pattern.endswith(">") and not pattern.startswith("</"):
                                                            closing_tag = "</" + pattern[1:]
                                                            if closing_tag in turn_content:
                                                                contains_thinking = True
                                                                self.logger.info(f"Found thinking token pattern: {pattern} with closing tag {closing_tag}")
                                                                break
                                                        # For other formats, do a simpler check
                                                        else:
                                                            contains_thinking = True
                                                            self.logger.info(f"Found thinking token pattern: {pattern}")
                                                            break
                                                
                                                # Force processing in debug mode only for the first turn of the first answer
                                                force_debug = self.debug and turn_idx == 0 and choice_idx == 0 and len(processed_answers) == 0
                                                
                                                if contains_thinking or force_debug:
                                                    if force_debug and not contains_thinking:
                                                        self.logger.info(f"Debug mode enabled - forcing processing of first turn")
                                                    else:
                                                        self.logger.info(f"Found thinking tokens in turn {turn_idx} of choice {choice_idx}")
                                                    
                                                    has_thinking_tokens = True
                                                    
                                                    # For debugging, log a snippet of content before processing
                                                    max_log_length = 100
                                                    self.logger.info(f"Original content (first {max_log_length} chars): " + 
                                                                     turn_content[:max_log_length] + "...")
                                                    
                                                    # Apply post-processing with full functionality
                                                    # We'll use both regex and model-based cleaning if possible
                                                    processed_turn = self.apply_reasoning_postprocessing(turn_content)
                                                    
                                                    # Log sample of processed result
                                                    self.logger.info(f"Processed content (first {max_log_length} chars): " + 
                                                                     processed_turn[:max_log_length] + "...")
                                                    
                                                    # Update the processed answer
                                                    processed_ans["choices"][choice_idx]["turns"][turn_idx] = processed_turn
                                                else:
                                                    # No thinking tokens, keep original
                                                    self.logger.debug(f"No thinking tokens in turn {turn_idx}, keeping original")
                                
                                processed_answers.append(processed_ans)
                                
                                # In debug mode, log the first answer's before/after for verification
                                if self.debug and processed_ans == processed_answers[0]:
                                    self.logger.info(f"Debug information for first answer (ID: {processed_ans.get('question_id', 'unknown')})")
                                    if has_thinking_tokens:
                                        self.logger.info("Post-processing was applied to this answer")
                                    else:
                                        self.logger.info("No thinking tokens found, original content was preserved")
                            
                            # Write processed answers back to file
                            backup_file = self.answer_dir / f"{model_id}.original.jsonl"
                            if not backup_file.exists():
                                import shutil
                                shutil.copy(answer_file, backup_file)
                                self.logger.info(f"Backed up original answers to {backup_file}")
                            
                            with open(answer_file, "w") as f:
                                for ans in processed_answers:
                                    f.write(json.dumps(ans) + "\n")
                            
                            self.logger.info(f"Wrote post-processed answers back to {answer_file}")

                            # IMPORTANT: Flush writes to disk to ensure they're visible to subsequent reads
                            try:
                                import os
                                os.fsync(f.fileno())
                                self.logger.info("Flushed file writes to disk")
                            except Exception as flush_err:
                                self.logger.warning(f"Error flushing file to disk: {str(flush_err)}")
                        else:
                            self.logger.info(f"No thinking tokens found in {answer_file}, skipping post-processing")
                else:
                    self.logger.warning("Could not load post-processing model, continuing with original responses")
            except Exception as e:
                self.logger.error(f"Error during post-processing: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                self.logger.warning("Continuing with original unprocessed responses")

        # Load data for evaluation - minimal memory footprint by loading directly from disk
        questions = load_questions(self.question_file, None, None)
        if self.debug:
            questions = questions[:10]
            self.logger.info(f"Debug mode: using 10 examples")

        # Reload model answers from disk to ensure we're using the processed versions
        self.logger.info("Loading model answers from disk (including processed versions)")
        model_answers = load_model_answers(self.answer_dir)
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
            # Phase 1: Generate responses with main model
            # This minimizes memory usage during the main LLM generation phase
            self.logger.info("Generating responses...")
            generation_results = self.generate_responses(model)

            # If not primary rank, return None early
            if generation_results is None:
                return None
                
            # Release VLLM resources before starting post-processing (if possible)
            if hasattr(model, "release_resources") and callable(model.release_resources):
                self.logger.info("Releasing VLLM model resources to free VRAM...")
                model.release_resources()

            # Phase 2: Load model answers from disk for post-processing
            # This completely separates generation from post-processing,
            # avoiding any need to keep the main model and post-processing model in memory simultaneously
            if self.reasoning_postproc and self.postproc_model is not None:
                self.logger.info("Applying reasoning post-processing to MTBench responses...")
                try:
                    # Read answer files instead of keeping them in memory
                    model_id = generation_results["model_id"]
                    answer_file = self.answer_dir / f"{model_id}.jsonl"
                    
                    if answer_file.exists():
                        # Load answers from disk
                        self.logger.info(f"Loading answers from {answer_file}")
                        with open(answer_file, "r") as f:
                            answers = [json.loads(line) for line in f]
                            
                        # Ensure the post-processing model is loaded only once we need it
                        if self._ensure_postproc_model_loaded():
                            # Process each answer file
                            processed_answers = []
                            for ans in answers:
                                # We need to target the nested content inside the choices array
                                processed_ans = ans.copy()  # Make a copy to avoid modifying the original
                                
                                # Process each choice's turn content specifically
                                if "choices" in processed_ans:
                                    for choice_idx, choice in enumerate(processed_ans["choices"]):
                                        if "turns" in choice:
                                            # Process each turn in the choice
                                            for turn_idx, turn_content in enumerate(choice["turns"]):
                                                # Apply post-processing to the turn content (a string)
                                                processed_turn = self.apply_reasoning_postprocessing(turn_content)
                                                # Replace the original content with the processed version
                                                processed_ans["choices"][choice_idx]["turns"][turn_idx] = processed_turn
                                                
                                                # Log the first turn's before/after for debugging
                                                if choice_idx == 0 and turn_idx == 0:
                                                    self.logger.info(f"Processed turn content from: {turn_content[:50]}... to: {processed_turn[:50]}...")
                                
                                processed_answers.append(processed_ans)
                        else:
                            self.logger.warning("Could not load post-processing model, using original answers")
                            processed_answers = answers
                            
                        # Create a temporary file for the processed answers
                        temp_answer_file = self.answer_dir / f"{model_id}.processed.jsonl"
                        with open(temp_answer_file, "w") as f:
                            for ans in processed_answers:
                                f.write(json.dumps(ans) + "\n")
                                
                        # Backup original answers
                        backup_file = self.answer_dir / f"{model_id}.original.jsonl"
                        if not backup_file.exists():  # Only backup if not already backed up
                            import shutil
                            shutil.copy(answer_file, backup_file)
                            self.logger.info(f"Backed up original answers to {backup_file}")
                            
                        # Replace original with processed
                        import os
                        os.replace(temp_answer_file, answer_file)
                        self.logger.info(f"Replaced original answers with processed answers")
                        
                        # Log example for debugging
                        if answers and processed_answers and len(answers) > 0 and len(processed_answers) > 0:
                            try:
                                # Extract first response from original and processed answers
                                orig_turns = answers[0]["choices"][0]["turns"] if "choices" in answers[0] else []
                                proc_turns = processed_answers[0]["choices"][0]["turns"] if "choices" in processed_answers[0] else []
                                
                                if orig_turns and proc_turns:
                                    self.logger.info(f"Original response example: {orig_turns[0][:100]}...")
                                    self.logger.info(f"Processed response example: {proc_turns[0][:100]}...")
                            except (KeyError, IndexError) as e:
                                self.logger.warning(f"Could not extract example responses: {e}")
                    else:
                        self.logger.warning(f"Answer file {answer_file} does not exist, skipping post-processing")
                except Exception as e:
                    self.logger.error(f"Error during post-processing: {str(e)}")
                    self.logger.warning("Using original unprocessed responses")
                    import traceback
                    self.logger.error(traceback.format_exc())
            else:
                self.logger.info("Reasoning post-processing is disabled or not available")

            # Phase 3: Evaluate responses
            self.logger.info("Evaluating responses...")
            evaluation_results = self.evaluate_responses(generation_results)
            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}
