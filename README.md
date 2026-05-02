# LiZIP: An Auto-Regressive Compression Framework for LiDAR Point Clouds

[![CI](https://github.com/HWUDLabAIRoboticsResearch/LiZIP/actions/workflows/ci.yml/badge.svg)](https://github.com/HWUDLabAIRoboticsResearch/LiZIP/actions/workflows/ci.yml)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)

This repo is the official code base for **LiZIP**, a lightweight near-lossless compression framework for LiDAR point clouds based on neural predictive coding. The project website is [here](https://hwudlabairoboticsresearch.github.io/LiZIP/).

<div align="center">
  <img src="benchmark/gifs/error_heatmap_lizip.gif" width="30%" alt="Reconstruction Error (LiZIP)" style="margin: 0 1.5%;"/>
  <img src="benchmark/gifs/error_heatmap_laszip.gif" width="30%" alt="Reconstruction Error (LASzip)" style="margin: 0 1.5%;"/>
  <img src="benchmark/gifs/error_heatmap_draco.gif" width="30%" alt="Reconstruction Error (Draco)" style="margin: 0 1.5%;"/>
</div>

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

## Key Results

- **7.5%–14.8%** smaller files than the industry-standard LASzip across NuScenes and Argoverse.
- **8.8%–11.3%** smaller than Google Draco (24-bit precision baseline) while keeping reconstruction error ≤ 0.017 mm vs. Draco's 0.033–0.070 mm.
- **38%–48%** smaller than GZip — a 3.8× compression ratio on a typical NuScenes frame (683.9 KB raw → 184.8 KB).
- Runs entirely on **CPU** (~75 ms/frame, C++ backend with AVX2 + OpenMP). No GPU required at inference time.
- On **NVIDIA Jetson** (AGX Orin), CUDA voxel sort + TensorRT inference drops total pipeline time significantly as elaborated in [Benchmark Results](#benchmark-results).
- Generalises to the unseen **Argoverse** dataset without retraining.

---

## Benchmark Results

### NuScenes

| Method | Total Time (ms) | Time vs LiZIP | Time vs LASzip | Size (KB) | Size vs LiZIP | Size vs LASzip | Max Error (mm) |
|---|---|---|---|---|---|---|---|
| LiZIP (C++, zlib, CUDA) | 54.2 ± 0.9 | — (Baseline) | +88.9% | 197.87 ± 28.48 | — (Baseline) | −1.3% | 0.0107 |
| LiZIP (C++, zlib, i7) | 75.0 ± 13.0 | +38.4% | +161.3% | 198.2 ± 29.0 | +0.2% | −1.1% | 0.0100 |
| LiZIP (C++, lzma, CUDA) | 150.5 ± 7.1 | +177.7% | +424.7% | 185.39 ± 28.03 | −6.3% | −7.5% | 0.0107 |
| LiZIP (C++, lzma, i7) | 192.0 ± 27.0 | +254.2% | +568.6% | 185.4 ± 28.0 | −6.3% | −7.5% | 0.0100 |
| LASzip | 28.7 ± 2.5 | −47.1% | — (Baseline) | 200.47 ± 16.49 | +1.3% | — (Baseline) | 0.0108 |
| Draco | 52.7 ± 2.9 | −2.8% | +83.6% | 203.32 ± 19.72 | +2.8% | +1.4% | 0.0259 |
| GZip | 105.0 ± 22.0 | +93.7% | +265.9% | 355.9 ± 42.0 | +79.9% | +77.5% | 0 |

### Argoverse

| Method | Total Time (ms) | Time vs LiZIP | Time vs LASzip | Size (KB) | Size vs LiZIP | Size vs LASzip | Max Error (mm) |
|---|---|---|---|---|---|---|---|
| LiZIP (C++, zlib, CUDA) | 66.8 ± 0.8 | — (Baseline) | +19.9% | 626.13 ± 7.86 | — (Baseline) | −11.4% | 0.0169 |
| LiZIP (C++, zlib, i7) | 191.0 ± 22.0 | +185.9% | +242.9% | 625.8 ± 8.0 | −0.1% | −11.4% | 0.0170 |
| LiZIP (C++, lzma, CUDA) | 228.3 ± 5.9 | +241.8% | +309.7% | 603.65 ± 6.57 | −3.6% | −14.6% | 0.0169 |
| LiZIP (C++, lzma, i7) | 415.0 ± 45.0 | +521.3% | +645.1% | 602.3 ± 6.0 | −3.8% | −14.7% | 0.0170 |
| LASzip | 55.7 ± 0.7 | −16.6% | — (Baseline) | 706.54 ± 6.30 | +12.8% | — (Baseline) | 0.0175 |
| Draco | 119.7 ± 7.7 | +79.2% | +114.9% | 679.24 ± 6.26 | +8.5% | −3.9% | 0.0545 |
| GZip | 62.0 ± 10.0 | −7.2% | +11.3% | 973.5 ± 13.0 | +55.5% | +37.8% | 0 |

> CUDA results measured on NVIDIA Jetson AGX Orin. i7 results measured on Intel Core i7 (CPU-only). LiZIP zlib is used as the time/size baseline throughout.

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

## Jetson CUDA Acceleration (TensorRT)

LiZIP auto-detects NVIDIA Jetson hardware at runtime and switches to the TensorRT + CUDA path. Pre-built models for the AGX Orin are in `models/jetson/`. If you need to rebuild or re-export:

### 1. Export the PyTorch model to ONNX

```bash
python scripts/export_onnx.py
```

### 2. Compile the TensorRT engine

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/jetson/mlp_c3_h256.onnx \
    --saveEngine=models/jetson/mlp_c3_h256.engine \
    --fp16 \
    --minShapes=input:1x9 \
    --optShapes=input:512x9 \
    --maxShapes=input:2048x9
```

### 3. Build the Jetson C++ engine

```bash
cd src/cpp/jetson
make
```

This produces `src/cpp/jetson/lizip`, which is used automatically when running on Jetson.

### 4. Run

No extra flags needed, the Jetson binary and engine are selected automatically:

```bash
python main.py encode input.bin output.lizip --mode cpp
python main.py benchmark --dataset nuscenes --frames 100 --mode cpp
```

To force a specific engine:

```bash
python main.py encode input.bin output.lizip --mode cpp --model models/jetson/mlp_c3_h256.engine
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
