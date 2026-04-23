# Progressive Compression of ResNet: An 8.33× Reduction With 90.5% Accuracy Retention for Edge Deployment

[![IEEE Access](https://img.shields.io/badge/IEEE%20Access-Vol.14%202026-blue)](https://doi.org/10.1109/ACCESS.2026.3682854)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2026.3682854-blue)](https://doi.org/10.1109/ACCESS.2026.3682854)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?logo=researchgate&logoColor=white)](https://www.researchgate.net/publication/403718114_Progressive_Compression_of_ResNet_An_833_Reduction_with_905_Accuracy_Retention_for_Edge_Deployment)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**Tu Tien Ngo**\*, **Nguyen Bich**\*, Pham The Bao, Son Ngoc Truong, Huan Minh Vo, Khoa Van Pham, Kyeong-Sik Min

\* Equal contribution

- ¹ School of Electrical Engineering, Kookmin University, Seoul 02707, Republic of Korea *(Tu Tien Ngo, Nguyen Bich, Kyeong-Sik Min)*
- ² Information Science Faculty, Saigon University, Ho Chi Minh City 70000, Vietnam *(Pham The Bao)*
- ³ Faculty of Electrical and Electronics Engineering, HCMC University of Technology and Engineering, Ho Chi Minh City 70000, Vietnam *(Son Ngoc Truong, Huan Minh Vo, Khoa Van Pham)*

📄 **[Read the full paper on IEEE Xplore](https://doi.org/10.1109/ACCESS.2026.3682854)** | 📑 **[ResearchGate](https://www.researchgate.net/publication/403718114_Progressive_Compression_of_ResNet_An_833_Reduction_with_905_Accuracy_Retention_for_Edge_Deployment)**

*Received 26 March 2026; accepted 7 April 2026; published 10 April 2026. IEEE Access, vol. 14, pp. 56900–56918.*

---

## Abstract

Deploying deep convolutional neural networks on edge devices remains challenging due to limited on-chip memory capacity, strict power budgets, and the high energy cost of off-chip DRAM access. This work presents a systematic, hardware-aware framework for designing compact CNN architectures inspired by the ResNet family. The proposed design integrates depthwise–separable convolutions, low-rank channel decomposition, and a zero-parameter identity shortcut that mitigates rank bottleneck degradation. The resulting **ResNet for Edge** is trained entirely from scratch. Compared with ResNet-18, it reduces the parameter count by **8.33×** (11.69M → 1.404M) and the computational cost by **42.3×** (1819M → 43M FLOPs) while retaining **90.5%** of the baseline's Top-1 accuracy on ImageNet-1K. On the NVIDIA Jetson Orin Nano (15W mode, FP16), the model reaches up to **1415 FPS** and consumes only **327.2 J** to process the entire 50,000-image ImageNet validation set—achieving the best accuracy-per-energy efficiency among all eight compared models.

---

## Key Results on ImageNet-1K

Evaluated on NVIDIA Jetson Orin Nano, 15W mode, FP16, batch size 128, mean ± std over 5 runs.

| Model | Params (M) | FLOPs (M) | Top-1 (%) | Top-5 (%) | FPS (BS=128) | Energy (J) |
|---|---|---|---|---|---|---|
| **ResNet for Edge (Ours)** | **1.404** | **43** | **62.922** | **83.908** | **1415.2** | **327.2** |
| ShuffleNetV2 0.5× | 1.367 | 43 | 59.974 | 81.314 | 1399.5 | 320.0 |
| MobileNetV3-Small | 2.543 | 60 | 67.330 | 87.280 | 1142.6 | 398.3 |
| MobileNetV4-Small | 3.774 | 190 | 73.752 | 91.422 | 964.4 | 469.2 |
| ResNet-18 | 11.690 | 1819 | 69.544 | 89.090 | 542.1 | 959.3 |
| SqueezeNet v1.1 | 1.235 | 349 | 57.864 | 80.426 | 571.5 | 819.5 |
| iFormer-T | 2.886 | 538 | 74.056 | 92.202 | 363.0 | 1190.4 |
| StarNet-S1 | 2.869 | 425 | 73.544 | 91.492 | 282.3 | 1492.3 |

Our ResNet for Edge achieves the **best accuracy-per-energy efficiency** (0.1923 %/J) among all compared models — 2.6% higher than ShuffleNetV2 (0.1874 %/J) and 17.8% lower energy than MobileNetV3-Small (398.3 J).

---

## Method Overview

**ResNet for Edge** combines three complementary, hardware-friendly compression operators applied progressively across network stages:

1. **Depthwise Separable Convolution (DSC)** — factorizes spatial filtering and channel mixing, reducing spatial computation cost. Each 3×3 conv is replaced by a depthwise 3×3 followed by a pointwise 1×1.
2. **Low-Rank Decomposition (LR)** — replaces dense 1×1 pointwise convolutions with two smaller bottleneck layers (Cᵢₙ → r → Cₒᵤₜ), reducing channel-mixing cost when r ≪ min(Cᵢₙ, Cₒᵤₜ).
3. **Identity Shortcut for Low-Rank (SC)** — a zero-parameter residual connection that bypasses the low-rank bottleneck to preserve information flow when Cᵢₙ = Cₒᵤₜ and stride = 1.

These operators act on complementary dimensions of the parameter space and produce regular, structured patterns that map efficiently to commodity edge hardware — no NAS, no attention modules, no custom operators. The rank parameter r is the single knob controlling the accuracy–efficiency trade-off, enabling continuous model-size tuning across deployment targets.

**Architecture:** Four-stage ResNet-inspired structure with channels [16, 40, 120, 192], ranks [0, 16, 32, 64], lightweight 3×3 stem (stride 2), and an expanded 1×1 classification head (192 → 1000). Total: **1.404M parameters**, **43M FLOPs**, **2.8 MB in FP16** (fits entirely in on-chip SRAM).

---

## Ablation Results

Validated across CIFAR-100 (ResNet-18 backbone) and Tiny-ImageNet (ResNet-34 backbone):

- **SC shortcut** provides the largest single accuracy gain: +9 pp on CIFAR-100 and +9.4 pp on Tiny-ImageNet at rank r=8 with zero additional parameters.
- **Knowledge Distillation (KD)** adds a consistent +0.86–1.31% improvement when combined with SC on CIFAR-100, but degrades ImageNet results due to the capacity gap between teacher and ultra-compact student.
- Best CIFAR-100 result: **75.96% Top-1** (DSC+LR+KD+SC, rank=128, ResNet-18 baseline: 79.81%).
- Best Tiny-ImageNet result: **65.10% Top-1** (DSC+LR+KD+SC, rank=256, ResNet-34 baseline: 72.99%).

---

## Index Terms

Depthwise separable convolution, low-rank decomposition, edge computing, energy efficiency, shortcut connection, image classification, model compression, CNN, edge deployment, NVIDIA Jetson.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{ngo2026progressive,
  title     = {Progressive Compression of {ResNet}: An $8.33\times$ Reduction With $90.5\%$ Accuracy Retention for Edge Deployment},
  author    = {Ngo, Tu Tien and Bich, Nguyen and Bao, Pham The and Truong, Son Ngoc and Vo, Huan Minh and {Van Pham}, Khoa and Min, Kyeong-Sik},
  journal   = {IEEE Access},
  volume    = {14},
  pages     = {56900--56918},
  year      = {2026},
  doi       = {10.1109/ACCESS.2026.3682854}
}
```

---

## Related Work

Key works compared or referenced in this paper:

| Model | Reference | Venue |
|---|---|---|
| ResNet | He et al., "Deep Residual Learning for Image Recognition" | CVPR 2016 |
| SqueezeNet | Iandola et al., "SqueezeNet: AlexNet-level accuracy with 50× fewer parameters" | arXiv 2016 |
| MobileNets | Howard et al., "MobileNets: Efficient CNNs for mobile vision applications" | arXiv 2017 |
| MobileNetV2 | Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" | CVPR 2018 |
| ShuffleNetV2 | Ma et al., "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" | ECCV 2018 |
| MobileNetV3 | Howard et al., "Searching for MobileNetV3" | ICCV 2019 |
| FlyNet | Guerra & Drummond, "FlyNet: Max it, excite it, quantize it" | BMVC 2022 |
| GhostNetV3 | Liu et al., "GhostNetV3: Exploring the training strategies for compact models" | arXiv 2024 |
| MobileNetV4 | Qin et al., "MobileNetV4: Universal Models for the Mobile Ecosystem" | ECCV 2024 |
| StarNet | Ma et al., "Rewrite the Stars" | CVPR 2024 |
| iFormer-T | Zheng, "iFormer: Integrating ConvNet and Transformer for Mobile Application" | arXiv 2025 |

---

## Acknowledgment

This work was supported by the National Research Foundation of Korea (NRF) under Grant RS-2024-00441473, RS-2024-00401234, RS-2024-00406006, and RS-2025-12872969. CAD tools were technically supported by IC Design Education Center (IDEC), Daejeon, South Korea.

---

## License

This work is licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/).
