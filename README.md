# Reasoning Length and Model Size

This project studies how **reasoning length** and **model size** affect performance on reasoning benchmarks.

We evaluate language models under different **step-by-step reasoning conditions** and measure performance on two datasets: **GSM8K** and **DROP**.

---

## Team Members

Weihao Li, Shijie Chen, Katie Shao,	Yunfei Ge,	Di Hu

---

## Project Structure

```text
project/
│
├── gsm8k/
│   ├── eval_gsm8k_prompt.py
│   └── eval_gsm8k_step.py
│
├── drop/
│   ├── eval_drop.py
│   ├── drop_scoring.py
│   └── test.py
│
├── results/
│   ├── small/
│   └── large/
│
└── README.md

Directory Description
Folder	Description
gsm8k/	Evaluation scripts for GSM8K math reasoning dataset
drop/	Evaluation scripts and scoring functions for DROP
results/	Stores evaluation results
README.md	Project documentation
Datasets

We evaluate models on two reasoning benchmarks:

GSM8K

Grade school math word problems

Requires multi-step arithmetic reasoning

DROP

Reading comprehension with discrete reasoning

Requires operations like:

counting

comparison

arithmetic

Reasoning Conditions

Each model is evaluated under four prompting conditions controlling the number of reasoning steps.

Condition	Target Steps	Description
direct	0	Direct answer, no reasoning
short	2	Short reasoning
medium	4	Medium reasoning
long	8	Long reasoning
Example Prompt Format
Step 1: ...
Step 2: ...
Final Answer: ...
Evaluation Metric

We use Exact Match (EM).

A prediction is correct if:

normalize(prediction) == normalize(gold_answer)

Normalization removes:

punctuation

articles

casing differences

For DROP, multiple gold answers are allowed. A prediction is correct if it matches any normalized gold answer.

Running Experiments
1) Install Dependencies
pip install torch transformers datasets tqdm
2) Run DROP Experiments

Example command:

python drop/eval_drop.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --condition short \
  --split validation \
  --max-samples 200 \
  --temperature 0 \
  --device auto
Arguments
Argument	Description
--model	HuggingFace model name
--condition	Reasoning condition (direct, short, medium, long)
--split	Dataset split
--max-samples	Number of examples to evaluate
--temperature	Decoding temperature
--device	cpu / cuda / auto
Models Evaluated
Small / Medium Models

Qwen2.5-1.5B

Qwen2.5-7B-Instruct

Llama-3.2-1B-Instruct

Llama-3.2-3B-Instruct

Results

Results are saved in the results/ directory:

results/
├── small/
└── large/
Project Goal

This project aims to understand:

How chain-of-thought length affects model performance

Whether larger models benefit more from longer reasoning

Differences between math reasoning (GSM8K) and reading reasoning (DROP)

Future Work

Possible extensions include:

Testing larger models (e.g., 13B+)

Adding more reasoning benchmarks

Evaluating self-consistency decoding

Analyzing reasoning step quality
