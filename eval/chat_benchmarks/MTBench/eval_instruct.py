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
            debug: If True, run in debug mode on 2 samples
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

    def get_model_answers(self, model: LM, model_id: str, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate model answers for all questions."""
        # Initialize tracking structures
        all_convs = [[] for _ in questions]
        all_choices = [{"index": 0, "turns": []} for _ in questions]
        all_answers = []  # Store the complete answer objects

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
                    # Store complete answer objects for post-processing
                    all_answers.append(ans_json)

        return {"choices": all_choices, "answers": all_answers, "questions": questions}

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate responses for MTBench questions.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing model responses and metadata, or None for non-primary ranks
        """
        # Load questions
        questions = load_questions(self.question_file, self.config.question_begin, self.config.question_end)

        if self.debug:
            questions = questions[:2]
            self.logger.info("Debug mode: using first 2 questions")

        # Shuffle questions for better load balancing
        random.shuffle(questions)

        # Generate answers
        response_data = self.get_model_answers(model=model, model_id=model.model_identifier, questions=questions)

        # Return None early for non-primary ranks if compute() returned None
        if response_data is None:
            return None

        # Include all response data for post-processing
        return {
            "model_id": model.model_identifier,
            "response_data": response_data
        }

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate model responses using GPT-4 judge.

        Args:
            results: Dictionary containing model identifier and post-processed response data

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        # Handle None result from non-primary ranks
        if results is None:
            return None

        # Save post-processed answers to file first
        answer_file = self.answer_dir / f"{results['model_id']}.jsonl"
        
        # If file exists, remove it to avoid duplicate entries
        if answer_file.exists():
            answer_file.unlink()
            
        if "response_data" in results and "answers" in results["response_data"]:
            self.logger.info("Writing post-processed answers to file")
            for ans_json in results["response_data"]["answers"]:
                with open(answer_file, "a") as f:
                    f.write(json.dumps(ans_json) + "\n")
                    
        # Load data for evaluation
        if "response_data" in results and "questions" in results["response_data"]:
            questions = results["response_data"]["questions"]
            if self.debug:
                self.logger.info(f"Debug mode: using {len(questions)} examples")
        else:
            questions = load_questions(self.question_file, None, None)
            if self.debug:
                questions = questions[:2]
                self.logger.info(f"Debug mode: using 2 examples")

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
            # Generate responses
            self.logger.info("Generating responses...")
            generation_results = self.generate_responses(model)

            # If not primary rank, return None early
            if generation_results is None:
                return None

            # Apply reasoning post-processing if enabled
            if self.reasoning_postproc and self.postproc_model is not None:
                self.logger.info("Applying reasoning post-processing to MTBench responses...")
                try:
                    processed_results = self.apply_reasoning_postprocessing(generation_results)
                    self.logger.info("Post-processing complete")
                    
                    # Log some examples of before/after processing for debugging
                    if "response_data" in generation_results and "response_data" in processed_results:
                        orig_choices = generation_results["response_data"].get("choices", [])
                        proc_choices = processed_results["response_data"].get("choices", [])
                        
                        if orig_choices and proc_choices and len(orig_choices) > 0 and len(proc_choices) > 0:
                            # Log first example only
                            orig_turn = orig_choices[0].get("turns", [])[0] if orig_choices[0].get("turns") else "No turn data"
                            proc_turn = proc_choices[0].get("turns", [])[0] if proc_choices[0].get("turns") else "No turn data"
                            
                            self.logger.info(f"Original response example: {orig_turn[:100]}...")
                            self.logger.info(f"Processed response example: {proc_turn[:100]}...")
                    
                    generation_results = processed_results
                except Exception as e:
                    self.logger.error(f"Error during post-processing: {str(e)}")
                    self.logger.warning("Using original unprocessed responses")
            else:
                self.logger.info("Reasoning post-processing is disabled or not available")

            # Evaluate responses
            self.logger.info("Evaluating responses...")
            evaluation_results = self.evaluate_responses(generation_results)
            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}
