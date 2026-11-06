# checkpoints

Small (trainable) heads bundled with the repo:

| file | size | purpose |
| --- | ---: | --- |
| `e2vbase_head.pt` | ~0.78 MB | 5-class head (angry/happy/neutral/sad/surprise) over emotion2vec_base embeddings |
| `cnscd_multilabel_head.pt` | ~0.78 MB | CNSCED 7-label multi-label head (5 emotions + fear + aroused) |

Both are MLP heads over the frozen 768-d `emotion2vec_base` utterance embedding.

**The backbones are NOT in this repo** (too large). Download them with
`python scripts/download_models.py` (puts them under `models/`), or train your own.
