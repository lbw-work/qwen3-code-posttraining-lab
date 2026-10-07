---
license: apache-2.0
language:
- en
base_model: Qwen/Qwen3-1.7B-Base
datasets:
- nvidia/OpenCodeInstruct
pipeline_tag: text-generation
library_name: transformers
tags:
- qwen3
- python
- code-generation
- post-training
- supervised-fine-tuning
---

# Qwen3-1.7B Python Code Full SFT

This model is a learning-project checkpoint produced by full supervised fine-tuning of [`Qwen/Qwen3-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-1.7B-Base) for Python function completion. It is a **base-style code completion model**: the experiment uses a raw Python prompt, not a chat conversation. Source code, frozen data and per-task evaluation are in the [Qwen3 Code Post-Training Lab](https://github.com/lbw-work/qwen3-code-posttraining-lab).

本模型属于代码后训练学习实验。它用于观察同一 Base 模型经过 Full SFT 后，Python 函数题的通过率如何变化；不代表生产级代码助手。

## Training

- Base revision: `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- Data: 26,805 training and 1,386 validation examples filtered from a fixed 100,000-row shard of [NVIDIA OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct). The source dataset is CC BY 4.0; attribution and the exact source revision are recorded in the [data lock](https://github.com/lbw-work/qwen3-code-posttraining-lab/blob/main/data/sft.lock.json).
- Single RTX 4090; 1 epoch, 1,676 optimizer steps, effective batch size 16, bf16, max sequence length 1,024, learning rate `2e-5`. Loss is computed only on the code completion, not the prompt.
- Best validation loss: `0.14692`. The final model was selected by validation loss, not by public benchmark score.

## Evaluation

| Benchmark | Base | This Full SFT model |
| --- | ---: | ---: |
| HumanEval+ v0.1.10 | 31/164 (18.9%) | **67/164 (40.9%)** |
| MBPP+ v0.2.0 | 214/378 (56.6%) | **229/378 (60.6%)** |

Strict pass@1 requires both the original and Plus tests to pass. All stages use the same frozen tasks, raw EvalPlus prompt, one greedy completion per task and at most 512 new tokens. Original completions are judged without code sanitization in a restricted Docker container. [Per-task results and the evaluation contract](https://github.com/lbw-work/qwen3-code-posttraining-lab/blob/main/docs/sft-comparison.md) are public.

## Load and generate

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "Eternity5551/Qwen3-1.7B-Python-Code-Full-SFT"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="auto", device_map="auto").eval()

prompt = '"""\nWrite a Python function double_even(nums) that doubles only even integers.\n"""\n\n'
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, do_sample=False, max_new_tokens=512,
                        pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

The repository contains a standard sharded Transformers model and tokenizer; `from_pretrained` loads it directly.

## Limitations and credits

The evaluation covers public Python function benchmarks only. Prompt overlap filtering cannot exclude every semantic duplicate, and two official benchmark reference solutions failed their own tests in this environment; all models were still scored on the full 164/378 tasks. Do not treat these numbers as a guarantee on unseen software engineering work. Base model: Qwen team, Apache 2.0. Training data: NVIDIA OpenCodeInstruct, CC BY 4.0.
