# Reasoning Post-Processing

This feature allows for the removal of internal reasoning chains, self-referential language, and thinking tokens from model responses before they are evaluated. This can be useful for benchmarks where you want to evaluate the final answer quality without penalizing models for showing their work.

## Usage

### Command Line Options

Two command-line flags have been added to enable and configure reasoning post-processing:

```bash
python -m eval.eval --model hf --tasks MTBench --model_args pretrained=<model_name> \
    --reasoning-postproc \
    --reasoning-postproc-model "Qwen/Qwen2.5-7B-Instruct"
```

- `--reasoning-postproc`: Enable reasoning post-processing (flag, no value)
- `--reasoning-postproc-model`: Specify the model to use for post-processing (defaults to "Qwen/Qwen2.5-7B-Instruct")

### Configuration File

You can also specify these options in your YAML configuration file:

```yaml
tasks:
  - task_name: MTBench
    batch_size: auto
reasoning_postproc: true
reasoning_postproc_model: "Qwen/Qwen2.5-7B-Instruct"
```

## How It Works

When reasoning post-processing is enabled:

1. Each benchmark receives a post-processing model instance during initialization
2. After generating responses but before evaluation, the benchmark applies post-processing to clean up the responses
3. The post-processing consists of two steps:
   - Pattern-based cleaning using regex to remove common thinking token formats (e.g., `<thinking>...</thinking>`)
   - Optional model-based processing to identify and remove reasoning chains and self-referential language

## Model-based Processing

The model-based processing uses the specified LLM to clean up the text. The model is instructed to:

- Copy the entire response
- Remove internal reasoning chains that are repetitive or that didn't lead to the final answer
- Remove self-referential talk like "wait", "stop", "I'm not sure", "let's think", etc.
- Keep the final answer and any essential context

## Extending for New Benchmarks

If you're implementing a new benchmark, the post-processing will be automatically available. The base `BaseBenchmark` class handles initializing the post-processing model and provides a utility method `apply_reasoning_postprocessing` that your benchmark can use.

The post-processing is automatically applied in the `run_benchmark` method of `BaseBenchmark`, after generating responses but before evaluation. If you override this method, make sure to call the super implementation or include the post-processing step manually.

## Example

Here's an example of a response before and after post-processing:

### Before
```
I'll solve this step by step. <thinking>First, I need to calculate the derivative of f(x) = x^2. The derivative is f'(x) = 2x.</thinking> The answer is f'(x) = 2x.
```

### After
```
I'll solve this step by step. The answer is f'(x) = 2x.
```

## Notes

- If post-processing fails for any reason, the original unprocessed responses will be used
- The post-processing model is loaded in bfloat16 precision to minimize memory usage
- Post-processing adds some overhead to evaluation time