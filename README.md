# NOVA-SGL: Novelty-Aware Social Graph Learning for Recommendation

This repository provides the official implementation of **NOVA-SGL**, a novelty-aware social graph learning framework for recommender systems.

NOVA-SGL integrates user–item graph propagation, user–user social graph learning, personalized novelty estimation, novelty-aware gating, popularity-bias mitigation, and social contrastive learning.

## Main Components

- User–item graph propagation
- Social graph propagation
- Collaborative–social representation fusion
- Personalized global/social novelty estimation
- User-specific novelty gate
- Popularity-bias mitigation
- Symmetric social contrastive learning
- RecBole-compatible training and evaluation

## Repository Structure

```text
nova-sgl/
├── novasgl/
│   └── models/
│       └── novasgl.py
├── scripts/
│   ├── register_novasgl.py
│   ├── run_novasgl_ml100k.py
│   ├── run_novasgl_lastfm_social.py
│   └── run_baselines.py
├── configs/
├── docs/
├── results/
└── data/
```
## Installation

Create and activate a Python environment:
```text
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:
```text
pip install -r requirements.txt
```

For CUDA-enabled PyTorch, install the CUDA wheel matching your system. For example:

```text
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```
## Register NOVA-SGL in RecBole

Before running experiments, register the model inside the installed RecBole package:
```text
python scripts/register_novasgl.py
```

## Dataset Preparation

For the full social version, use a dataset with both user–item interactions and user–user social links.

For LastFM, place the processed atomic files as:
```text
data/
└── lastfm/
    ├── lastfm.inter
    ├── lastfm.item
    └── lastfm.net
```
Expected fields:
```text
lastfm.inter: user_id:token, artist_id:token
lastfm.net: source_id:token, target_id:token
```
Run NOVA-SGL on LastFM
```
python scripts/run_novasgl_lastfm_social.py
```

Citation
If you use this repository, please cite our paper:
```text
@article{nova_sgl_2026,
  title={NOVA-SGL: Novelty-Aware Social Graph Learning for Recommendation},
  author={Zare, Gholamreza and others},
  journal={Under Review},
  year={2026}
}
```