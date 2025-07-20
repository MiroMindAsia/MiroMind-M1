# Supervised Fine-tuning

We use [LLama-Factory](https://github.com/hiyouga/LLaMA-Factory) to run our SFT experiment. Below are the setup steps

### 1. Install LLama-Factory `v0.9.1` (specific commit).

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout 1176cd67790a3c21aeaace495399efbf177ab51f
```

### 2. Add the custom dataset.
Edit `LLaMA-Factory/data/dataset_info.json` to include:
```python
{
  "miromind-m1-sft-data": {
    "file_name": "PATH_TO_YOUR_OWN_DATASET",
    "formatting": "alpaca",
    "columns": {
    "query": "question",
    "response": "response"
    }
}
}
```
Replace `PATH_TO_YOUR_OWN_DATASET` with your dataset location.

### 3. Register the custom chat template
Edit `LLaMA-Factory/src/llamafactory/data/template.py` and add:
```python
register_template(
	name="qwen25_r1_sft",
	...
)
```

### 4. Start training 
Run:
```bash
FORCE_TORCHRUN=1 NNODES=1 llamafactory-cli train examples/train_full/qwen2.5_full_sft.yaml
```
Replace `PATH_TO_YOUR_CONFIG.yaml` with your actual config file path.
