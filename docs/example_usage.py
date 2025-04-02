"""
Example showing how to use the reasoning post-processing utility.

This example demonstrates how a benchmark would use the post-processing
functionality in its generate_responses method.
"""

import logging
from typing import Dict, Any, List, Optional

from lm_eval.api.model import LM
from eval.task import BaseBenchmark
from eval.utils.reasoning_postproc import postprocess_object


class ExampleBenchmark(BaseBenchmark):
    """Example benchmark that uses reasoning post-processing."""
    
    def __init__(
        self, 
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        reasoning_postproc: bool = False,
        reasoning_postproc_model: str = "Qwen/Qwen2.5-7B-Instruct",
    ):
        super().__init__(
            logger=logger, 
            system_instruction=system_instruction,
            reasoning_postproc=reasoning_postproc,
            reasoning_postproc_model=reasoning_postproc_model,
        )
        
    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """Generate responses for the benchmark questions."""
        # Generate responses (in a real benchmark this would call the model)
        responses = [
            "I'll solve this step by step. <thinking>First, I need to calculate the derivative of f(x) = x^2. The derivative is f'(x) = 2x.</thinking> The answer is f'(x) = 2x.",
            "Let me think about this. <|begin_of_thought|>We need to find the square root of 144. 12 * 12 = 144, so the square root is 12.</|end_of_thought|> The square root of 144 is 12."
        ]
        
        results = {
            "responses": responses,
            "metadata": {"benchmark": "Example"}
        }
        
        # In a real benchmark, post-processing would be applied in the run_benchmark method
        # which calls this generate_responses method and then applies post-processing to the results
        
        # For this example, we'll manually show how to apply post-processing
        if self.reasoning_postproc and self.postproc_model is not None:
            self.logger.info("Applying reasoning post-processing to example responses")
            try:
                # Use the utility function to post-process the results
                processed_results = self.apply_reasoning_postprocessing(results)
                self.logger.info(f"Post-processing complete")
                
                # Print original and processed responses for comparison
                self.logger.info("Original responses:")
                for resp in results["responses"]:
                    self.logger.info(f"  - {resp}")
                    
                self.logger.info("Processed responses:")
                for resp in processed_results["responses"]:
                    self.logger.info(f"  - {resp}")
                
                return processed_results
            except Exception as e:
                self.logger.error(f"Error during post-processing: {str(e)}")
                return results
        else:
            self.logger.info("Reasoning post-processing is disabled")
            return results
    
    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the model's responses."""
        # In a real benchmark, this would implement evaluation logic
        return {"score": 0.5}


# Example usage (not executed when the file is imported)
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("example")
    
    # Create the benchmark with post-processing enabled
    benchmark = ExampleBenchmark(
        logger=logger,
        reasoning_postproc=True,
    )
    
    # This would normally be run by the framework
    result = benchmark.run_benchmark(None)  # Pass None as we don't use the model in this example
    
    logger.info(f"Evaluation result: {result}")