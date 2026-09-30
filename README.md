# C-BDIA: Fast and Consistent Video Editing via Bidirectional Integration and Caching

Official implementation of **C-BDIA**, a training-free acceleration framework for diffusion-based video editing.

**Huiwen Luo, Guoqiang Zhang, Zhidong Li**

[Paper (Coming Soon)](#) | [Project Page (Coming Soon)](#)

---

## Overview

C-BDIA accelerates diffusion-based video editing by combining feature caching with bidirectional integration correction.

While feature caching reduces redundant neural network evaluations, aggressive caching can introduce approximation errors and degrade editing quality.

C-BDIA addresses this limitation by introducing a bidirectional correction at cache-hit steps, using neighboring latent states to compensate for cache-induced deviations without additional denoising-network evaluations.

Our approach is built upon FateZero and requires no additional training.

### Highlights

- Training-free acceleration for diffusion-based video editing.
- Bidirectional correction for cache-induced approximation errors.
- No additional neural network evaluations at cache-hit steps.
- Improved editing quality under aggressive feature caching.
- Compatible with the FateZero video editing framework.

## Results

We evaluate C-BDIA on DAVIS and in-the-wild videos across style, object, background, and attribute editing.

### Quantitative Comparison

The following results compare C-BDIA with cached DDIM (C-DDIM) at comparable inference costs.

| Method | Speedup | CLIP | LPIPS | SSIM | PSNR |
|---|---:|---:|---:|---:|---:|
| C-DDIM (0.14) | 2.37x | 26.32 | 0.5799 | 0.5584 | 16.36 |
| C-BDIA (0.14) | 2.41x | 33.34 | 0.4411 | 0.5855 | 16.16 |
| C-DDIM (0.3) | 3.62x | 25.25 | 0.7007 | 0.5131 | 13.87 |
| C-BDIA (0.3) | 3.62x | 30.08 | 0.6157 | 0.5471 | 15.97 |

C-BDIA demonstrates improved editing quality under aggressive caching while maintaining comparable inference efficiency.

### Qualitative Results

Visual comparisons of C-DDIM and C-BDIA.

| Input | C-DDIM | C-BDIA |
|:---:|:---:|:---:|
| Coming Soon | Coming Soon | Coming Soon |

## Installation

Our implementation is based on the original FateZero repository.

```bash
git clone https://github.com/luolora35/C-BDIA.git
cd C-BDIA

conda create -n cbdia python=3.8
conda activate cbdia

pip install -r requirements.txt
```

Download the Stable Diffusion v1.4 checkpoint and place it under:

```text
ckpt/stable-diffusion-v1-4/
```

The pretrained checkpoint is available from:

https://huggingface.co/CompVis/stable-diffusion-v1-4

## Inference

C-BDIA supports configurable sampling steps, cache thresholds, and bidirectional correction strength.

Key experimental parameters:

| Parameter | Description |
|---|---|
| T | Number of sampling steps |
| delta | Cache threshold |
| gamma | Bidirectional correction strength |

Our experiments evaluate the original DDIM with T = 40, 30, 20, and 12.

For aggressive caching, we use delta = 0.30 and gamma = 0.5.

Detailed inference commands and example configurations will be provided with the code release.

## Acknowledgements

This work is built upon [FateZero](https://github.com/ChenyangQiQi/FateZero).

We sincerely thank the FateZero authors for releasing their code and pretrained models.

We also acknowledge the contributions of [TeaCache](https://github.com/ali-vilab/TeaCache) and the original BDIA method, which inspired the caching and numerical integration components of this work.

## Citation

If you find our work useful, please consider citing C-BDIA.

Citation information will be updated upon publication.

Please also cite the original FateZero paper:

```bibtex
@inproceedings{qi2023fatezero,
  title={FateZero: Fusing Attentions for Zero-shot Text-based Video Editing},
  author={Qi, Chenyang and Cun, Xiaodong and Zhang, Yong and Lei, Chenyang and Wang, Xintao and Shan, Ying and Chen, Qifeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

