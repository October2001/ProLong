# ProLong

This repository contains code for the **ACL'2024 Oral** paper "**[Long Context is Not Long at All: A Prospector of Long-Dependency Data for Large Language Models](https://arxiv.org/abs/2405.17915)**".

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
datasets==2.15.0
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

## Key parameters
* `chunk_size` - The chunk size to be used for processing the data, here we use 128
* `window_size` - The maximum window size to be considered, here we use 32768
* `dlt_ppl_threshold` - The threshold to be used for filter delta perplexity, here we use 0.1
* `single_ppl_batch_size` - The batch size to be used for calculating single perplexity
* `pair_ppl_batch_size` - The batch size to be used for calculating pair perplexity
* `sample_size` - The sample size to be used when calculating pair perplexity, if sample size is set to -1, then sampling strategy will not be used, all pairs will be calculated

## ProLong Test Set
* The toy test set constructed in the paper is relatively small. Subsequently, we will release a larger ProLong test set with a broader source to assist users in selecting hyperparameters based on their experimental settings.

## Citation

If you find this repository helpful, please consider citing the following paper:

```bib
@article{chen2024long,
  title={Long Context is Not Long at All: A Prospector of Long-Dependency Data for Large Language Models},
  author={Chen, Longze and Liu, Ziqiang and He, Wanwei and Li, Yunshui and Luo, Run and Yang, Min},
  journal={arXiv preprint arXiv:2405.17915},
  year={2024}
}
```

## Contact
<!-- email -->

If you have any questions, feel free to contact us at `lz.chen2@siat.ac.cn` or `ww.he@siat.ac.cn`.
