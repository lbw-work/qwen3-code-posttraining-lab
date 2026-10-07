---
license: apache-2.0
language:
- en
base_model: Qwen/Qwen3-1.7B-Base
datasets:
- nvidia/OpenCodeInstruct
pipeline_tag: text-generation
library_name: peft
tags:
- qwen3
- python
- code-generation
- post-training
- lora
- supervised-fine-tuning
---

# Qwen3-1.7B Python Code LoRA SFT

This is a **LoRA adapter**, not a standalone model. Load it on the pinned [`Qwen/Qwen3-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-1.7B-Base) revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. It belongs to the [Qwen3 Code Post-Training Lab](https://github.com/lbw-work/qwen3-code-posttraining-lab), a hands-on project comparing code post-training methods under one evaluation contract.

这是用于 Python 函数补全的 LoRA SFT 适配器。输入应是原始代码题干，不是聊天消息；只下载本仓库的适配器文件无法独立推理，还需下载锁定版本的 Base 权重。

## Training

- Same 26,805 training and 1,386 validation examples as the project's Full SFT model, filtered from a fixed shard of [NVIDIA OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) (CC BY 4.0). Exact source revision and attribution: [data lock](https://github.com/lbw-work/qwen3-code-posttraining-lab/blob/main/data/sft.lock.json).
- Single RTX 4090; 1 epoch, 1,676 optimizer steps, effective batch size 16, bf16, max sequence length 1,024.
- LoRA rank 16, alpha 32, dropout 0.05, attention and MLP linear layers; learning rate `1e-4`. Loss is computed only on the code completion.
- Best validation loss: `0.14723`. The adapter was selected by validation loss, not benchmark score.

## Evaluation

| Benchmark | Base | Full SFT | This LoRA SFT adapter |
| --- | ---: | ---: | ---: |
| HumanEval+ v0.1.10 | 31/164 (18.9%) | 67/164 (40.9%) | **82/164 (50.0%)** |
| MBPP+ v0.2.0 | 214/378 (56.6%) | 229/378 (60.6%) | **237/378 (62.7%)** |

Strict pass@1 requires both original and Plus tests to pass. The same frozen tasks, raw prompt, greedy decoding, 512-token cap and restricted Docker judge were used for all three models. The different SFT learning rates and optimizers mean this result should not be attributed solely to the LoRA update mechanism. [Per-task results and evaluation details](https://github.com/lbw-work/qwen3-code-posttraining-lab/blob/main/docs/sft-comparison.md) are public.

## Load and generate

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B-Base",
    revision="ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    dtype="auto",
    device_map="auto",
)
adapter = "Eternity5551/Qwen3-1.7B-Python-Code-LoRA-SFT"
model = PeftModel.from_pretrained(base, adapter).eval()
tokenizer = AutoTokenizer.from_pretrained(adapter)

prompt = '"""\nWrite a Python function double_even(nums) that doubles only even integers.\n"""\n\n'
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, do_sample=False, max_new_tokens=512,
                        pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## Limitations and credits

These are public Python function benchmarks and cannot guarantee performance on unseen code. Filtering by exact word spans cannot rule out all semantic overlap. Base model: Qwen team, Apache 2.0. Training data: NVIDIA OpenCodeInstruct, CC BY 4.0. The adapter weights were verified byte-for-byte against the training output before upload.
