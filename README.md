# Databricks Ray vLLM Inference

This project contains the `databricks-ray-vllm.py` notebook for running distributed LLM inference on Databricks using Ray and vLLM.

**Original Author:** [Puneet Jain](https://www.linkedin.com/in/puneetjain159/)

## Usage

1.  **Import to Databricks:**
    Import `databricks-ray-vllm.ipynb` into your Databricks workspace. It is a standard Jupyter Notebook.

2.  **Configuration:**
    -   **Model:** Qwen/Qwen3-4B-Instruct-2507 (Non-FP8 for A10 stability)
    -   **Infrastructure:** Serverless GPU (AWS A10s recommended)
    -   **Scaling:** Adjust `num_instances` and `@ray_launch(gpus=...)` to scale up.

## Architecture

How the distributed inference pipeline works on Databricks Serverless GPUs:

```ascii
                                    +-----------------------+
                                    |   Databricks Driver   |
                                    |      (Notebook)       |
                                    +-----------+-----------+
                                                |
                                       @ray_launch(gpus=5)
                                                |
                                                v
                                    +-----------------------+
                                    |    Ray Head Node      |
                                    |   (Serverless GPU)    |
                                    |   - Ray Controller    |
                                    |   - Data Scheduler    |
                                    +------+----+----+------+
                                           |    |    |
                        +------------------+    |    +------------------+
                        |                       |                       |
            +-----------v-----------+ +---------v-----------+ +---------v-----------+
            |     Ray Worker 1      | |     Ray Worker 2    | |     Ray Worker 5    |
            |   (Serverless GPU)    | |   (Serverless GPU)  | |   (Serverless GPU)  |
            +-----------------------+ +---------------------+ +---------------------+
            |      vLLM Engine      | |     vLLM Engine     | |     vLLM Engine     |
            |   (Qwen Model Copy)   | |  (Qwen Model Copy)  | |  (Qwen Model Copy)  |
            +-----------+-----------+ +---------+-----------+ +---------+-----------+
                        |                       |                       |
                        v                       v                       v
                 [Batch of 32]           [Batch of 32]           [Batch of 32]
```

## Key Files

-   `databricks-ray-vllm.ipynb`: The main inference notebook source code.
-   `qwen_inference.py`: A clean, simplified script version for testing.
-   `requirements.txt`: Python dependencies.

## Notes

-   **A10 Compatibility:** Uses `bfloat16` dtype for A10 GPUs.
-   **Concurrency:** Ensure `num_instances` equals `gpus` for 1:1 mapping (max throughput).
