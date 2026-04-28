# LiZIP: An Auto-Regressive Compression Framework for LiDAR Point Clouds

[![CI](https://github.com/HWUDLabAIRoboticsResearch/LiZIP/actions/workflows/ci.yml/badge.svg)](https://github.com/HWUDLabAIRoboticsResearch/LiZIP/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

This repo is the official code base for **LiZIP**, a lightweight near-lossless compression framework for LiDAR point clouds based on neural predictive coding. The project website is [here](https://hwudlabairoboticsresearch.github.io/LiZIP/).

<div align="center">
  <img src="benchmark/gifs/error_heatmap_lizip.gif" width="30%" alt="Reconstruction Error (LiZIP)" style="margin: 0 1.5%;"/>
  <img src="benchmark/gifs/error_heatmap_laszip.gif" width="30%" alt="Reconstruction Error (LASzip)" style="margin: 0 1.5%;"/>
  <img src="benchmark/gifs/error_heatmap_draco.gif" width="30%" alt="Reconstruction Error (Draco)" style="margin: 0 1.5%;"/>
</div>

---

## Key Results

- **7.5%–14.8%** smaller files than the industry-standard LASzip across NuScenes and Argoverse.
- **8.8%–11.3%** smaller than Google Draco (24-bit precision baseline) while keeping reconstruction error ≤ 0.017 mm vs. Draco's 0.033–0.070 mm.
- **38%–48%** smaller than GZip — a 3.8× compression ratio on a typical NuScenes frame (683.9 KB raw → 184.8 KB).
- Runs entirely on **CPU** (~75 ms/frame, C++ backend with AVX2 + OpenMP). No GPU required at inference time.
- Generalises to the unseen **Argoverse** dataset without retraining.

---

## Getting Started

### Prerequisites

- Python 3.9+
- A C++ compiler with OpenMP support (for the C++ backend)

### Installation

```bash
git clone https://github.com/HWUDLabAIRoboticsResearch/LiZIP
cd LiZIP
pip install -r requirements.txt
```

---

## Usage

### Encode a point cloud

```bash
# Python backend (default model: mlp_c3_h256)
python main.py encode input.bin output.lizip

# C++ backend — faster, recommended
python main.py encode input.bin output.lizip --mode cpp

# Best compression ratio (lzma)
python main.py encode input.bin output.lizip --mode cpp --compression lzma

# Custom model variant
python main.py encode input.bin output.lizip --model models/grid_search/mlp_c8_h1024.bin --mode cpp
```

### Decode

```bash
python main.py decode output.lizip reconstructed.bin --mode cpp
```

### Compare original vs. reconstructed

```bash
python src/utils/compare.py input.bin reconstructed.bin
```

Reports Chamfer distance, Hausdorff distance, and p95/p99 nearest-neighbour error in mm.

### Benchmark against Draco, LASzip and GZip

```bash
python main.py benchmark --dataset nuscenes --frames 100 --mode dual
```

---

## Visualisation Tools

Generate the GIFs shown above from any NuScenes `.bin` file:

```bash
# Rotating reconstruction error heatmap
python src/utils/make_error_heatmap_gif.py data/nuScenes/LIDAR_TOP/<frame>.bin --out error_heatmap.gif

# Residual distribution tightening animation
python src/utils/make_residual_dist_gif.py data/nuScenes/LIDAR_TOP/<frame>.bin --out residual_dist.gif
```

Both scripts require `imageio` in addition to the standard requirements:

```bash
pip install imageio
```

---

## Citation

```bibtex
@misc{LiZIP,
      title={LiZIP: An Auto-Regressive Compression Framework for LiDAR Point Clouds},
      author={Aditya Shibu and Kayvan Karim and Claudio Zito},
      year={2026},
      eprint={2603.23162},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.23162},
}
```
