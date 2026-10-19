# Label Authority in Speech Emotion Recognition

Code for the paper

> M. Giordano, S. Giacomelli, C. Rinaldi, F. Graziosi,
> **"Label Authority in Speech Emotion Recognition: Intended and Perceived Ground Truth in Emozionalmente."**

Ground truth in speech emotion recognition (SER) is usually treated as a single
categorical fact. Emozionalmente, an acted Italian corpus, gives every clip both
the emotion the speaker was asked to produce and five listener judgements. This
repository compares three **label authorities** on the same clips and the same
speaker-independent split:

| Authority | Target | Question it answers |
|---|---|---|
| Expressed | one-hot elicitation label | Is the requested acted condition recoverable from speech? |
| Recognized plurality | most frequent of five votes, ties to the lowest class index | Which single class wins under the stated aggregation rule? |
| Recognized distribution | seven-class vector of vote proportions | How should probability be spread across the alternatives listeners chose? |

Each authority is learned by shallow probes (`linear`, `dense`, `agg_dense`) on
frozen XLS-R 300M, HuBERT Large and WavLM Large hidden states: 63 probe
configurations per target, five runs each, 945 training runs in total.

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps

python -m label_authority.labels       # 1. label authorities from the listener votes
python -m label_authority.features     # 2. frozen hidden states (GPU, high RAM)
python -m label_authority.train        # 3. 189 probe configurations x 5 runs (GPU)
python -m label_authority.analysis     # 4. every table and in-text number of the paper
python -m label_authority.figures      # 5. Figures 1-5
```

[`docs/STUDY.md`](docs/STUDY.md) explains the study design, the repository
layout, every step, and where each paper table and figure comes from.

## Data

The Emozionalmente corpus is **not** distributed with this repository. Obtain it
from its authors (F. Catania et al., *IEEE/ACM TASLP* 33, 2025) under its own
license, and place it at `data/emozionalmente/` (or set
`LABEL_AUTHORITY_DATASET`):

```
data/emozionalmente/
├── audio/*.wav
└── metadata/
    ├── samples.csv        # file_name, actor, emotion_expressed, ...
    └── evaluations.csv    # file_name, evaluator, audio_quality, emotion_recognized
```

Extracted features (`features/`) and all outputs (`results/`) are generated
locally and are ignored by git.

## Citation

```bibtex
@inproceedings{giordano2026labelauthority,
  author    = {Giordano, Marco and Giacomelli, Stefano and Rinaldi, Claudia and Graziosi, Fabio},
  title     = {Label Authority in Speech Emotion Recognition: Intended and Perceived Ground Truth in Emozionalmente},
  year      = {2026}
}
```

## License

Copyright © 2026 Marco Giordano.

This work is licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
You may share and adapt it for non-commercial purposes, provided you give
appropriate credit and indicate if changes were made. See [`LICENSE`](LICENSE).

The license covers the code in this repository only. Emozionalmente and the
pretrained backbones (XLS-R, HuBERT, WavLM) remain under their own licenses.

## Acknowledgements

This work was partially supported by the European Union under the Italian
National Recovery and Resilience Plan (NRRP) of NextGenerationEU, partnership on
"Telecommunications of the Future" (PE00000001 – program "RESTART").
