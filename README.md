Reasoning Length vs Model Size

When Does “Think More” Help?

This project studies how reasoning length affects the performance of language models on reasoning benchmarks.
We evaluate models under different step-length prompting conditions and analyze the relationship between:

reasoning length

accuracy

computational cost

Project Structure
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
direct	0	direct answer, no reasoning
short	2	short reasoning
medium	4	medium reasoning
long	8	long reasoning

Example prompt format:

Step 1: ...
Step 2: ...
Final Answer: ...
Evaluation Metric

We use Exact Match (EM).

Prediction is correct if:

normalize(prediction) == normalize(gold_answer)

Normalization removes:

punctuation

articles

casing differences

For DROP, multiple gold answers are allowed.

Running Experiments
1 Install Dependencies
pip install torch transformers datasets tqdm
2 Running DROP Experiments

Example command:

python drop/eval_drop.py \
--model Qwen/Qwen2.5-1.5B-Instruct \
--condition short \
--split validation \
--max-samples 200 \
--temperature 0 \
--device auto

Arguments:

Argument	Description
model	HuggingFace model name
condition	reasoning condition
split	dataset split
max-samples	number of examples
temperature	decoding temperature
device	cpu / cuda / auto
