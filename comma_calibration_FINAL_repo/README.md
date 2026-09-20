# comma.ai Calibration Challenge — Multi-Scale Geometry Baseline

A compact solution for predicting the direction of vehicle travel in camera coordinates from dashcam video.

## Result

Leave-one-video-out validation on the five labeled challenge videos gives an aggregate **18.32% of the all-zero baseline** using the supplied challenge-style MSE normalization. This is a **local cross-validation result, not an official hidden leaderboard score**.

Per-video local scores:

| Held-out video | Error vs zero baseline |
|---|---:|
| 0 | 13.5% |
| 1 | 23.1% |
| 2 | 11.4% |
| 3 | 39.5% |
| 4 | 10.2% |
| Aggregate | **18.32%** |

## Approach

The dataset is very small, so this solution uses an interpretable geometry-first pipeline rather than a large neural model.

1. Detect Shi-Tomasi corners at quarter resolution.
2. Track them frame-to-frame with pyramidal Lucas-Kanade optical flow.
3. Robustly fit a translational flow model and derive an approximate focus of expansion (FOE).
4. Convert FOE into geometric pitch/yaw cues using the provided ~910 px focal length.
5. Add motion magnitude and fit-reliability features.
6. Build multi-timescale temporal features with Gaussian smoothing.
7. Learn a lightly regularized Ridge calibration from motion features to pitch/yaw.
8. Train on all five labeled videos and generate `5.txt` through `9.txt`.

## Why multi-scale temporal features?

Frame-level optical flow is noisy, while camera/vehicle calibration evolves much more slowly. Combining raw motion with 2/5/10/20/40/80-frame-scale smoothed signals improved leave-one-video-out generalization substantially over the first single-scale baseline.

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Extract features from all videos:

```bash
python src/motion_features.py /path/to/0.hevc /path/to/1.hevc ... /path/to/9.hevc
```

Keep the generated `*.hevc.feat.npy` files in the same data directory as the videos/labels, then validate:

```bash
python src/validate.py --data /path/to/data
```

Generate final predictions:

```bash
python src/train_predict.py --data /path/to/data --out submission
```

The submission directory contains `5.txt` through `9.txt`.

## Notes

- The 18.32% result is local cross-validation and must not be represented as an official comma.ai leaderboard score.
- Video 3 remains the hardest held-out scene, suggesting future work in explicit rotation compensation / essential-matrix ego-motion and road-region masking.
- The implementation intentionally prioritizes reproducibility and interpretability over model size.
