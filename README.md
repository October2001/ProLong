# ProLong

This repository contains code for the paper "**Long Context is Not Long at All: A Prospector of Long-Dependency Data for Large Language Models**".

## Abstract

Long-context modeling capabilities are important for large language models (LLMs) in various applications. However, directly training LLMs with long context windows is insufficient to enhance this capability since some training samples do not exhibit strong semantic dependencies across long contexts.
In this study, we propose a data mining framework **ProLong** that can assign each training sample with a long dependency score, which can be used to rank and filter samples that are more advantageous for enhancing long-context modeling abilities in LLM training. Specifically, we first use delta perplexity scores to measure the *Dependency Strength* between text segments in a given document. Then we refine this metric based on the *Dependency Distance* of these segments to incorporate spatial relationships across long-contexts. Final results are calibrated with a *Dependency Specificity* metric to prevent trivial dependencies introduced by repetitive patterns. Moreover, a random sampling approach is proposed to optimize the computational efficiency of ProLong. Comprehensive experiments on multiple benchmarks indicate that ProLong effectively identifies documents that carry long dependencies and LLMs trained on these documents exhibit significantly enhanced long-context modeling capabilities.

## Requirements
```
torch==2.1.1
transformers==4.36.0
tqdm==4.66.1
numpy==1.22.2
matplotlib==3.8.0
```

Install the required packages
```bash
pip install -r requirements.txt
```

## Data Preparation
The file to be processed should be in the following jsonl format:
```json
{"text": "This is the first sentence. This is the second sentence. This is the third sentence."}
{"text": "This is the first sentence. This is the second sentence. This is the third sentence."}
```

## Usage

### process single file with single process
```bash
bash scripts/run_single.sh
```

### process single file with single node and multiple processes
```bash
bash scripts/run_multiprocess.sh
```

### process multiple files with multiple nodes multiple processes
```bash
bash scripts/run_multinodes.sh
```


## Citation

If you find this repository helpful, please consider citing the following paper:

```
```
