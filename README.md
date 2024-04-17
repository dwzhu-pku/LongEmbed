# LongEmbed: Extending Embedding Models for Long Context Retrieval

This repository is the official implementation for the paper "LongEmbed: Extending Embedding Models for Long Context Retrieval"

This paper explores context window extension of existing embedding models, pushing the limit to 32k without requiring additional training. We have constructed the LongEmbed benchmark for long context retrieval, and investigates various methods to increase the input length of current embedding models.

![LongEmbed](assets/repo_figure.jpg)

## ⚡Released Models

To facilitate future research in long context embedding models, we release E5-Base-4k and E5-RoPE-Base. E5-Base-4k is further fine-tuned on E5-Base to support 4k context, while strictly preserving original behavior for inputs not exceeding 512 tokens. E5-RoPE-Base follows the same training procedure as E5-Base, except for the substitution of APE with RoPE. It is released to facilitate comparison between APE \& RoPE-Based embedding models.

| Model | Download Link |
| --- | --- |
| E5-Base-4k | Will be available before 20th April |
| E5-RoPE-Base | Will be available before 20th April |

## 🔍 Overview of LongEmbed

### Task Description

LongEmbed includes 4 real-world retrieval tasks curated from long-form QA and summarization. Note that for QA and summarization datasets, we use the questions and summaries as queries, respectively.

- [NarrativeQA](https://huggingface.co/datasets/narrativeqa): A QA dataset comprising long stories averaging 50,474 words and corresponding questions about specific content such as characters, events. We adopt the `test` set of the original dataset.
- [2WikiMultihopQA](https://huggingface.co/datasets/THUDM/LongBench/viewer/2wikimqa_e): A multi-hop QA dataset featuring questions with up to 5 hops, synthesized through manually designed templates to prevent shortcut solutions. We use the `test` split of the length-uniformly sampled version from [LongBench](https://huggingface.co/datasets/THUDM/LongBench).
- [QMSum](https://huggingface.co/datasets/tau/scrolls/blob/main/qmsum.zip): A query-based meeting summarization dataset that requires selecting and summarizing relevant segments of meetings in response to queries. We use the version processed by [SCROLLS](https://huggingface.co/datasets/tau/scrolls). Since its test set does not include ground truth summarizations, and its validation set only have 60 documents, which is too small for document retrieval, we include the `train` set in addition to the `validation` set.
- [SummScreenFD](https://huggingface.co/datasets/tau/scrolls/blob/main/summ_screen_fd.zip): A screenplay summarization dataset comprising pairs of TV series transcripts and human-written summaries. Similar to QMSum, its plot details are scattered throughout the transcript and must be integrated to form succinct descriptions in the summary. We use `validation` set of the version processed by [SCROLLS](https://huggingface.co/datasets/tau/scrolls).

We also include two synthetic tasks, namely needle and passkey retrieval. The former is tailored from the [Needle-in-a-Haystack Retrieval](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) for LLMs. The later is adopted from [Personalized Passkey Retrieval](https://huggingface.co/datasets/intfloat/personalized_passkey_retrieval), with slight change for the efficiency of evaluation. The advantage of synthetic data is that we can flexibly control context length and distribution of target information. For both tasks, we evaluate a broad context range of $\{0.25,0.5,1,2,4,8,16,32\}\times1024$ tokens. For each context length, we include 50 test samples, each comprising 1 query and 100 candidate documents.


### Task Statistics

| Dataset | Domain | # Queries | # Docs | Avg. Query Words | Avg. Doc Words |
|---------|--------|-----------|--------|------------------|----------------|
| NarrativeQA | Literature, File | 10,449 | 355 | 9 | 50,474 |
| QMSum | Meeting | 1,527 | 197 | 71 | 10,058 |
| 2WikimQA | Wikipedia | 300 | 300 | 12 | 6,132 |
| SummScreenFD | ScreenWriting | 336 | 336 | 102 | 5,582 |
| Passkey | Synthetic | 400 | 800 | 11 | - |
| Needle | Synthetic | 400 | 800 | 7 | - |

## 📈 Evaluation on the LongEmbed Benchmark

### Environment Setup

To replicate our results, follow these steps to download the code and necessary dependencies:
```
git clone https://github.com/dwzhu-pku/LongEmbed.git
cd LongEmbed
pip install -r requirements.txt
```
Note that we are still working on the integration of LongEmbed to MTEB. So for now, please manually add the `src/LEMB*` files to the `MTEB/mteb/tasks/Retrieval/eng/` folder and update the `MTEB/mteb/tasks/Retrieval/__init__.py` file accordingly. (please install MTEB in editable mode)

### Loading Data
LongEmbed contains six datasets: NarrativeQA, QMSum, 2WikiMultihopQA, SummScreenFD, Passkey, and Needle. Each dataset has three splits: corpus, queries, and qrels. The `corpus.jsonl` file contains the documents, the `queries.jsonl` file contains the queries, and the `qrels.jsonl` file describes the relevance. To spefic split of load each dataset, you may use:

```python
from datasets import load_dataset
# dataset_name in ["narrativeqa", "summ_screen_fd", "qmsum", "2wikimqa", "passkey", "needle"]
# split_name in ["corpus", "queries", "qrels"]
data_list = load_dataset(path="dwzhu/LongEmbed", name="dataset_name", split="split_name")
```

### Evaluation

The evaluation of LongEmbed can be easily conducted using MTEB. For the four real tasks, you can evaluate as follows:

```python
from tabulate import tabulate
from mteb import MTEB
retrieval_task_list = ["LEMBSummScreenFDRetrieval", "LEMBQMSumRetrieval","LEMBWikimQARetrieval","LEMBNarrativeQARetrieval"]
retrieval_task_results = []
evaluation = MTEB(tasks=retrieval_task_list)
results = evaluation.run(model,output_folder=args.output_dir, overwrite_results=True, batch_size=args.batch_size,verbosity=0)
for key, value in results.items():
	split = "test" if "test" in value else "validation"
	retrieval_task_results.append([key, value[split]["ndcg_at_1"], value[split]["ndcg_at_10"]])
	output_dict[key] = {"ndcg@1": value[split]["ndcg_at_1"], "ndcg@10": value[split]["ndcg_at_10"]}
print(tabulate(retrieval_task_results, headers=["Task", "NDCG@1", "NDCG@10"]))
```

For the two synthetic tasks, since we examine a broad context range of $ \{0.25,0.5,1,2,4,8,16,32\}\times1024 $ tokens, an additional parameter of `context_length` is required. You may evaluate as follows:

```python
from tabulate import tabulate
from mteb import MTEB
needle_passkey_score_list = []
for ctx_len in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
	print(f"Running task: NeedlesRetrieval, PasskeyRetrieval, context length: {ctx_len}")
	evaluation = MTEB(tasks=["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"])
	results = evaluation.run(model, context_length=ctx_len,overwrite_results=True,batch_size=args.batch_size)
	needle_passkey_score_list.append([ctx_len, results["LEMBNeedleRetrieval"]["test"]["ndcg_at_1"], results["LEMBPasskeyRetrieval"]["test"]["ndcg_at_1"]])
needle_passkey_score_list.append(["avg", sum([x[1] for x in needle_passkey_score_list])/len(context_length_list), sum([x[2] for x in needle_passkey_score_list])/len(context_length_list)])
print(tabulate(needle_passkey_score_list, headers=["Context Length", "Needle-ACC", "Passkey-ACC"]))
```

Our code snippet for evaluation can be found in `src/test_long_embed.py`. You may refer to the scripts in `scripts/run_long_embed.sh` to reproduce the results.



## 🌟 Citation
If you find this repo helpful, please cite our paper as follows:

```bibtex
```

