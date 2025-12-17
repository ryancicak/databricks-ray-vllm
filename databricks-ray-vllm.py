# Databricks notebook source
# MAGIC %md
# MAGIC # Ray vLLM Multi-Node Inference on Databricks (AWS)
# MAGIC
# MAGIC This notebook shows how to run large language model (LLM) inference at scale using Ray and vLLM on Databricks with AWS A10 GPUs. It uses Databricks serverless GPU infrastructure to automatically provision and manage resources for distributed inference.
# MAGIC
# MAGIC Original Author of the notebook: Puneet Jain https://www.linkedin.com/in/puneetjain159/
# MAGIC
# MAGIC Key steps:
# MAGIC - Install all required packages for Ray and vLLM distributed inference.
# MAGIC - Authenticate securely with Hugging Face for model access.
# MAGIC - Use Ray to launch and manage distributed workers across multiple GPUs.
# MAGIC - Run text inference pipelines with efficient batching and parallelism.
# MAGIC - Monitor Ray cluster resources to ensure optimal usage.
# MAGIC - Follow workspace policies for resource management, security, and cleanup.
# MAGIC
# MAGIC All compute is provisioned on-demand and cleaned up automatically, making it easy to scale up or down without manual cluster management.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Package Installation
# MAGIC
# MAGIC Installs all required packages for distributed Ray and vLLM inference on Databricks (AWS):
# MAGIC - Flash Attention (CUDA 12, PyTorch 2.6, Python 3.12, A10 GPU compatible)
# MAGIC - Databricks Connect for Spark integration
# MAGIC - Transformers <4.54.0, vLLM 0.8.5.post1
# MAGIC - OpenTelemetry Prometheus exporter, optree, hf_transfer, numpy
# MAGIC - Restarts Python for a clean environment
# MAGIC
# MAGIC All versions are pinned for compatibility and reproducibility.

# COMMAND ----------

# MAGIC %pip install --force-reinstall --no-cache-dir --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
# MAGIC %pip install databricks-connect<16
# MAGIC %pip install "transformers<4.54.0"
# MAGIC %pip install "vllm==0.8.5.post1"
# MAGIC %pip install  "opentelemetry.exporter.prometheus" 
# MAGIC %pip install  'optree>=0.13.0'
# MAGIC %pip install  hf_transfer
# MAGIC %pip install  "numpy==1.26.4"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Version Verification
# MAGIC **Purpose**: Verify package versions and enforce compatibility requirements
# MAGIC
# MAGIC **Key Actions**:
# MAGIC * Prints versions of torch, flash_attn, vllm, ray, transformers
# MAGIC * Asserts Ray version >= 2.47.1 for distributed features
# MAGIC
# MAGIC **Best Practices**: Version assertions prevent runtime errors in distributed setup
# MAGIC

# COMMAND ----------

from packaging.version import Version

import torch
import flash_attn
import vllm
import ray
import transformers

print(torch.__version__, flash_attn.__version__, vllm.__version__, ray.__version__, transformers.__version__)
assert Version(ray.__version__) >= Version("2.47.1"), (
    "Ray version must be at least 2.47.1"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hugging Face Authentication
# MAGIC **Display Name**: `===== LOGIN =====`
# MAGIC
# MAGIC **Purpose**: Authenticate with Hugging Face Hub for model access
# MAGIC
# MAGIC **Key Actions**: Uses `huggingface_hub.login()` for interactive authentication
# MAGIC
# MAGIC **Security**: Interactive login prevents hardcoding tokens (complies with FE workspace policy)
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,===== LOGIN =====
# Login to hugging face
from huggingface_hub import login

login()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: Ray Resource Reporting Function
# MAGIC
# MAGIC
# MAGIC **Purpose**: Utility function to inspect Ray cluster resources and debug distributed setups
# MAGIC
# MAGIC **Key Functions**:
# MAGIC * `print_ray_resources()`: Reports cluster resources and node details
# MAGIC * Shows GPU allocation per node with specific GPU IDs
# MAGIC * Graceful error handling for debugging
# MAGIC
# MAGIC **Usage**: Essential for verifying resource allocation matches expectations
# MAGIC

# COMMAND ----------

# DBTITLE 1,RAY REPORT
import json

def print_ray_resources():
    try:
        cluster_resources = ray.cluster_resources()
        print(f"Ray Cluster Resources: {json.dumps(cluster_resources, indent=2)}")

        nodes = ray.nodes()
        print(f"\nDetected {len(nodes)} Ray nodes:")
        for node in nodes:
            node_id = node.get("NodeID", "N/A")
            ip_address = node.get("NodeManagerAddress", "N/A")
            resources = node.get("Resources", {})
            num_gpus_ray = resources.get("GPU", 0.0) # GPU resource is typically a float

            print(f"  Node ID: {node_id}, IP: {ip_address}")
            print(f"    Ray-reported GPUs: {int(num_gpus_ray)}") # Convert float to int for display
            if "GPU" in resources and num_gpus_ray > 0:
                # If specific GPU IDs are reported, show them
                gpu_ids_on_node = [k for k, v in resources.items() if k.startswith("GPU_ID_")]
                if gpu_ids_on_node:
                    print(f"    Specific GPU IDs detected by Ray: {', '.join(gpu_ids_on_node)}")
            else:
                print(f"    No GPUs reported by Ray for this node.")

    except Exception as e:
        print(f"An error occurred while querying Ray cluster resources: {e}")

print_ray_resources()

# COMMAND ----------

# DBTITLE 1,===== MULI-NODE - CHAT =====

from typing import Any, Dict, List

import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM, SamplingParams

# def to_schema(item):
#     messages = [
#             {"role": "system", "content": "You are a bot that responds with haikus."},
#             {"role": "user", "content": item["item"]},
#         ]
#     return {'item': messages}


def scheduling_strategy_fn(tensor_parallel_size):
 
    pg = ray.util.placement_group(
        [{
            "GPU": 1,
            "CPU": 1
        }] * tensor_parallel_size,
        strategy="STRICT_PACK",
    )
    return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
        pg, placement_group_capture_child_tasks=True))





# COMMAND ----------

from serverless_gpu.ray import ray_launch 
import ray
import os
from packaging.version import Version

assert Version(ray.__version__) >= Version("2.22.0"), "Ray version must be at least 2.22.0"

class TaskRunner:
    def run():
        from typing import Any, Dict, List
        import numpy as np
        import ray
        from vllm import LLM, SamplingParams

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The future of AI is",
        ]
        ds = ray.data.from_items(1000*prompts) #this was 100 but if you make it higher than each node is assured to get prompts

        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

        tensor_parallel_size = 1
        num_instances = 5 

        class LLMPredictor:
            def __init__(self):
                # ✅ CHANGED: Switched to Non-FP8 model for stability on A10
                self.llm = LLM(
                    model="Qwen/Qwen3-4B-Instruct-2507", # Removed -FP8
                    tensor_parallel_size=1, 
                    dtype="bfloat16",             # Native for A10
                    trust_remote_code=True,       
                    gpu_memory_utilization=0.90,  
                    max_model_len=8192,           
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

        ds = ds.map_batches(
            LLMPredictor,
            concurrency=num_instances, 
            batch_size=32,
            num_gpus=1,
            num_cpus=12
        )

        outputs = ds.take(limit=10)
        for output in outputs:
            prompt = output["prompt"]
            generated_text = output["generated_text"]
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

@ray_launch(gpus=5, gpu_type='a10', remote=True)
def run() -> None:
    # os.environ['HF_TOKEN'] = 'hf_...' # Replace with your Hugging Face Token or use Databricks Secrets
    # Example: os.environ['HF_TOKEN'] = dbutils.secrets.get(scope="my-scope", key="hf-token")

    runner = TaskRunner.run()

import os 
os.environ['RAY_TEMP_DIR'] = '/tmp/ray'
run.distributed()