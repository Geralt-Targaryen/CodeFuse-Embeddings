## F2LLM

F2LLMs (Foundation-to-Feature Large Language Models) are foundation models directly finetuned on 6 million high-quality query-document pairs, striking a strong balance between model size, training cost, and embedding performance.

Model checkpoints are available at:

- [F2LLM 0.6B](https://huggingface.co/codefuse-ai/F2LLM-0.6B)
- [F2LLM 1.7B](https://huggingface.co/codefuse-ai/F2LLM-1.7B)
- [F2LLM 4B](https://huggingface.co/codefuse-ai/F2LLM-4B)

Training data is available at [F2LLM data](https://huggingface.co/datasets/codefuse-ai/F2LLM).

### Train

To reproduce the training of F2LLMs, please:

- Setup environment following `requirements.txt`. We note that transformers>=4.51.0 is required for training Qwen3 models.
- Download data and backbone models from Hugging Face (we use Qwen3 models).
- Run `tokenize_data_qwen.py` to tokenize the downloaded data
- Modify model path, data path, and other arguments in `configs/config.json`.
- Start training with `accelerate launch --config_file configs/accelerate_config.yaml run.py --config configs/config.json`.

For multi-node training, run on the main node:

```
accelerate launch --config_file configs/accelerate_config.yaml --num_machines N_NODE --num_processes N_PROCESSES --machine_rank 0 --main_process_ip MASTER_IP --main_process_port MASTER_PORT run.py --config configs/config.json
```

where N_NODE is the number of machines; N_PROCESSES is N_NODE\*8; MASTER_IP is the IP address of your master node, and MASTER_PORT is a port available on your machine (e.g. 6379).

On worker nodes, also run the above commmand but modify `machine_rank` accordingly.
