# comma-ai-calibration-challenge
Computer vision solution for the comma.ai Calibration Challenge using optical flow, focus-of-expansion geometry, temporal features, and robust regression to estimate vehicle pitch and yaw from dashcam video.

## About

This project is my solution to the comma.ai Calibration Challenge, which focuses on estimating the direction of vehicle travel in the camera frame from dashcam video.

The goal is to predict the pitch and yaw angle for every video frame. I built a geometry-first computer vision pipeline that combines optical flow, focus-of-expansion estimation, temporal motion features, and regression-based calibration.

The solution was developed and evaluated using the five labeled challenge videos with leave-one-video-out validation to measure how well the approach generalizes across different driving sequences.

### Approach

* Detect and track visual features between consecutive frames
* Estimate camera motion using optical flow
* Calculate a robust focus of expansion
* Convert motion geometry into pitch and yaw features
* Add temporal features to capture motion across multiple frames
* Train regression models using labeled driving sequences
* Generate predictions for the five unlabeled challenge videos

### Technologies

Python, OpenCV, NumPy, SciPy, Scikit-learn, Matplotlib

### Validation

The final version achieved an aggregate leave-one-video-out validation score of approximately **18.32% relative to the all-zero baseline** on the provided labeled videos.

This is a local cross-validation result and should not be interpreted as an official comma.ai leaderboard score.

### What I Learned

This project helped me explore computer vision and autonomous-driving problems from a systems perspective, including motion estimation, geometric reasoning, temporal modeling, robust validation, and failure analysis.

It also reinforced the importance of building simple, interpretable baselines before moving to more complex models, especially when working with small datasets.
