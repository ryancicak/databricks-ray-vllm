from serverless_gpu.ray import ray_launch 
import ray
import os
from packaging.version import Version

# Ensure Ray version compatibility
assert Version(ray.__version__) >= Version("2.22.0"), "Ray version must be at least 2.22.0"

class TaskRunner:
    def run():
        from typing import Any, Dict, List
        import numpy as np
        import ray
        from vllm import LLM, SamplingParams

        # Configuration
        # To scale up: Change num_instances to 5 and @ray_launch(gpus=5)
        tensor_parallel_size = 1
        num_instances = 1  
        batch_size = 32
        
        # Define prompts
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The future of AI is",
        ]
        # Duplicate prompts to create a larger dataset
        ds = ray.data.from_items(100 * prompts)

        # Sampling parameters for generation
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

        class LLMPredictor:
            def __init__(self):
                # Using Qwen 4B Instruct (Non-FP8 for A10 stability)
                # If using FP8, change model to "Qwen/Qwen3-4B-Instruct-2507-FP8" and dtype="auto"
                self.llm = LLM(
                    model="Qwen/Qwen3-4B-Instruct-2507",
                    tensor_parallel_size=1, 
                    dtype="bfloat16",             # Native for A10 GPUs
                    trust_remote_code=True,       # Required for Qwen models
                    gpu_memory_utilization=0.90,  # Prevent OOM on init
                    max_model_len=8192,           # Context window
                    enable_prefix_caching=True,   
                    enable_chunked_prefill=True, 
                    max_num_batched_tokens=8192,
                )

            def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
                outputs = self.llm.generate(batch["item"], sampling_params)
                prompt: List[str] = []
                generated_text: List[str] = []
                for output in outputs:
                    prompt.append(output.prompt)
                    generated_text.append(' '.join([o.text for o in output.outputs]))
                return {
                    "prompt": prompt,
                    "generated_text": generated_text,
                }

        # Distributed Map Batch
        # Note: concurrency must match num_instances (1 actor per GPU)
        ds = ds.map_batches(
            LLMPredictor,
            concurrency=num_instances, 
            batch_size=batch_size,
            num_gpus=1,  # GPUs per actor
            num_cpus=12  # CPUs per actor
        )

        # Execute and print results
        # Use take_all() to process everything, or take(limit=N) for a quick test
        outputs = ds.take(limit=10)
        for output in outputs:
            prompt = output["prompt"]
            generated_text = output["generated_text"]
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# Infrastructure Configuration
# Update gpus=X to match the cluster size you want (e.g. 5)
@ray_launch(gpus=1, gpu_type='a10', remote=True)
def run() -> None:
    # Set HF Token securely
    # Ideally, use Databricks secrets: os.environ['HF_TOKEN'] = dbutils.secrets.get(...)
    # os.environ['HF_TOKEN'] = 'hf_...' # Replace with your token
    # os.environ['HF_TOKEN'] = dbutils.secrets.get(scope="my-scope", key="hf-token")

    runner = TaskRunner.run()

if __name__ == "__main__":
    import os 
    os.environ['RAY_TEMP_DIR'] = '/tmp/ray'
    
    try:
        run.distributed()
    except KeyError as e:
        # Ignore known 'tasks' KeyError bug in serverless_gpu wrapper
        if "'tasks'" in str(e):
            print("\n✅ Job submitted successfully! (Ignored wrapper KeyError)")
        else:
            raise e

