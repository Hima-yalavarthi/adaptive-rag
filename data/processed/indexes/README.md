# FAISS indexes (generated locally)

Rebuild after cloning:

```bash
python -m src.retrieval.build_index --split validation
python -m src.retrieval.build_index --split train
# or
python -m src.retrieval.build_index --split all
```

Outputs:
- `hotpotqa_<split>.faiss` — vector index (Git-ignored; large)
- `hotpotqa_<split>_meta.jsonl` — chunk metadata aligned to vectors (Git-ignored)
- `hotpotqa_<split>_config.json` — model name, dimension, counts (tracked)
