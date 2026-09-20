# comma-ai-calibration-challenge

Computer vision solution for the comma.ai Calibration Challenge using optical flow, focus-of-expansion geometry, temporal features, and robust regression to estimate vehicle pitch and yaw from dashcam video.

## About

This project is my solution to the comma.ai Calibration Challenge, which focuses on estimating the direction of vehicle travel in the camera frame from dashcam video.

The goal is to predict the pitch and yaw angle for every frame of a driving video. I built a geometry-first computer vision pipeline that combines optical flow, focus-of-expansion estimation, temporal motion features, and regression-based calibration.

The solution was developed and evaluated using the five labeled challenge videos with leave-one-video-out validation to test how well the method generalizes across different driving sequences.

## Approach

The pipeline follows these steps:

1. Detect visual features in each frame.
2. Track features between consecutive frames using optical flow.
3. Filter unreliable and outlier motion tracks.
4. Estimate the focus of expansion from the observed motion.
5. Convert motion geometry into pitch and yaw related features.
6. Add temporal features to capture vehicle motion across multiple frames.
7. Train regression models using the labeled driving sequences.
8. Evaluate using leave-one-video-out validation.
9. Generate pitch and yaw predictions for the five unlabeled videos.

## Technologies

* Python
* OpenCV
* NumPy
* SciPy
* Scikit-learn
* Matplotlib

## Validation Results

The final model achieved an aggregate leave-one-video-out validation score of:

### **17.84% relative to the all-zero baseline**

Per-video validation results:

| Video         |      Error |
| ------------- | ---------: |
| Video 0       |     12.07% |
| Video 1       |     24.98% |
| Video 2       |     10.31% |
| Video 3       |     36.36% |
| Video 4       |     10.02% |
| **Aggregate** | **17.84%** |

Lower scores indicate better performance.

This is a local cross-validation result on the provided labeled videos and should not be interpreted as an official comma.ai leaderboard score.

## Results

### Validation Performance

![Validation Scores](results/plots/validation_scores.png)

### Example Pitch Prediction

![Video 0 Pitch](results/plots/video_0_pitch.png)

### Example Yaw Prediction

![Video 0 Yaw](results/plots/video_0_yaw.png)

Additional pitch and yaw plots for all labeled videos are available in:

```text
results/plots/
```

Sample dashcam frames are available in:

```text
results/frames/
```

## Project Structure

```text
comma-ai-calibration-challenge/
├── src/
│   ├── motion_features.py
│   ├── model.py
│   ├── validate.py
│   ├── train_predict.py
│   └── make_results.py
├── results/
│   ├── frames/
│   ├── plots/
│   ├── summary.txt
│   └── validation_scores.csv
├── submission/
│   ├── 5.txt
│   ├── 6.txt
│   ├── 7.txt
│   ├── 8.txt
│   └── 9.txt
├── requirements.txt
├── LICENSE
└── README.md
```

## Running the Project

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the challenge videos and labels inside a local `data/` directory.

Generate motion features:

```bash
python src/motion_features.py \
data/0.hevc \
data/1.hevc \
data/2.hevc \
data/3.hevc \
data/4.hevc \
data/5.hevc \
data/6.hevc \
data/7.hevc \
data/8.hevc \
data/9.hevc
```

Run leave-one-video-out validation:

```bash
python src/validate.py --data data
```

Generate result plots:

```bash
python src/make_results.py --data data --out results
```

Generate predictions for the unlabeled videos:

```bash
python src/train_predict.py --data data --out submission
```

The final predictions are written to:

```text
submission/5.txt
submission/6.txt
submission/7.txt
submission/8.txt
submission/9.txt
```

## What I Learned

This project helped me explore computer vision and autonomous-driving problems from a systems perspective, including motion estimation, geometric reasoning, temporal modeling, robust validation, and failure analysis.

One of the main lessons was the value of starting with an interpretable geometric baseline before moving toward more complex machine-learning approaches, especially when working with a small dataset.

The project also reinforced the importance of evaluating across complete driving sequences rather than relying only on training-set performance.

## Future Improvements

Possible improvements include:

* Essential-matrix based ego-motion estimation
* Better road-region and moving-object filtering
* More robust temporal modeling
* Camera-motion confidence estimation
* Runtime optimization for real-time execution
* Combining geometric features with learned representations

## Challenge

This project was built for the comma.ai Calibration Challenge and is intended as an independent technical exploration of motion estimation and camera calibration for autonomous-driving systems.
