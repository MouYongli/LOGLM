# On the translation from natural language to formal language via LLM prompting

## Python Environment Setup

1. conda environment
```
conda create --name=logicllm python=3.11.9
conda activate logicllm
```

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

2. jupyter lab and kernel
```
conda install -c conda-forge jupyterlab
```

## Datasets

1. AR-LSAT
2. FOLIO
3. LogicalDeduction
4. ProntoQA
5. ProofWriter
