# Weighted In-Context Learning（WICL）
This repo is for EMNLP 2023 Findings paper: 

Not All Demonstration Examples are Equally Beneficial: \
Reweighting Demonstration Examples for In-Context Learning 

## Requirments
Download this repo and enter working directory:

```
cd wicl
```

This repo need the following packages:

```
datasets==2.12.0
numpy==1.24.3
torch==1.12.1
tqdm==4.65.0
transformers==4.24.0
```

A suitable conda environment named `wicl` can be created and activated with: 

```
conda env create -f environment.yml
conda activate wicl
```

## Reproduce our experiments
Set proper hyper-parameters in `re_w.sh`, then:
```
bash re_w.sh
```

We utilize 5 different GPT-like causal language models released by fairseq (https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm), and the number of parameters of these models is 355M, 1.3B, 2.7B, 6.7B, 13B, respectively. For example, if you want to use GPT 6.7B, you can set `model_size = "6.7B"` in `re_w.sh`.

8 text classification are supported: 'sst2', 'mr', 'subj', 'agnews', 'cb', 'dbpedia', 'rte', 'boolq'  
For example, if you want to test on 'sst2' and 'mr', you can set `tasks = ('sst2' 'mr')` in `re_w.sh`.

<!-- `--indicator` is can be `MSP` or `validate_xx`(e.g. `validate_20`). For weight search, you can use MSP as a guidence under true few-shot setting. If an held-out validation set of xx examples as available, you can validate on this set as a guidence.  
`--beam_num` is a hyperparameter for beam serach  
`--re_weight_place` indicates how to add weights to ICL examples   -->




