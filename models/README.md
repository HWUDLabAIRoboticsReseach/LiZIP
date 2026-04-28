## Models

Nine pre-trained `PointPredictorMLP` variants are provided under `models/grid_search/`, covering three context sizes (k = 3, 5, 8) and three hidden dimensions (H = 256, 512, 1024). Each variant ships as both a PyTorch `.pth` checkpoint and a self-contained `.bin` binary (LIZM format) for the C++ engine.

| k | H    | Latency (s) | Size (KB) | Error (mm) |
|---|------|-------------|-----------|------------|
| **3** | **256** | **0.19** | **185.41** | **0.010** |
| 3 | 512  | 0.31        | 186.13    | 0.010      |
| 3 | 1024 | 1.06        | 185.42    | 0.010      |
| 5 | 256  | 0.18        | 186.17    | 0.010      |
| 5 | 512  | 0.33        | 184.88    | 0.010      |
| 5 | 1024 | 1.03        | 185.49    | 0.010      |
| 8 | 256  | 0.19        | 185.73    | 0.010      |
| 8 | 512  | 0.37        | 185.53    | 0.010      |
| 8 | 1024 | 1.23        | 184.50    | 0.010      |

The default model is `mlp_c3_h256` (bold above).