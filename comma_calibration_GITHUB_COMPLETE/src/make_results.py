from pathlib import Path
import argparse, sys, csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from model import make_features, fit, predict


def score(gt, pred):
    pred = np.nan_to_num(pred)
    return np.mean(np.nanmean((gt - pred) ** 2, axis=0))


def load_one(data, i):
    raw = np.load(data / f"{i}.hevc.feat.npy")
    X = make_features(raw)
    y = np.loadtxt(data / f"{i}.txt")
    n = min(len(X), len(y))
    X, y = X[:n], y[:n]
    valid = np.isfinite(y).all(axis=1)
    return X, y, valid


def save_sample_frame(video_path, out_path):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5.5))
    plt.imshow(frame)
    plt.axis('off')
    plt.title(video_path.name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--out', default=Path('results'), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plots = args.out / 'plots'
    frames = args.out / 'frames'
    plots.mkdir(exist_ok=True)
    frames.mkdir(exist_ok=True)

    videos = [load_one(args.data, i) for i in range(5)]
    fold_scores = []
    fold_rows = []

    for h in range(5):
        train_X, train_y = [], []
        for j, (X, y, valid) in enumerate(videos):
            if j == h:
                continue
            train_X.append(X[valid])
            train_y.append(y[valid])
        train_X = np.vstack(train_X)
        train_y = np.vstack(train_y)

        Xt, yt, valid = videos[h]
        pred = predict(fit(train_X, train_y), Xt)
        m = score(yt, pred)
        z = score(yt, np.zeros_like(yt))
        pct = 100 * m / z
        fold_scores.append((h, pct, m, z))
        fold_rows.append((m, z))

        t = np.arange(len(yt)) / 20.0
        for col, name in enumerate(['pitch', 'yaw']):
            plt.figure(figsize=(11, 4))
            plt.plot(t, yt[:, col], label='Ground truth', linewidth=1.5)
            plt.plot(t, pred[:, col], label='Prediction', linewidth=1.2)
            plt.xlabel('Time (s)')
            plt.ylabel(f'{name.title()} (rad)')
            plt.title(f'Video {h} — {name.title()} | LOVO {pct:.2f}%')
            plt.legend()
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(plots / f'video_{h}_{name}.png', dpi=160)
            plt.close()

        save_sample_frame(args.data / f'{h}.hevc', frames / f'video_{h}_sample.png')

    agg = 100 * np.mean([m for m, z in fold_rows]) / np.mean([z for m, z in fold_rows])

    # Summary bar chart
    labels = [f'Video {i}' for i, *_ in fold_scores]
    vals = [pct for _, pct, *_ in fold_scores]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, vals)
    plt.axhline(25, linestyle='--', linewidth=1.5, label='25% reference')
    plt.ylabel('Error vs all-zero baseline (%)')
    plt.title(f'Leave-One-Video-Out Validation — Aggregate {agg:.2f}%')
    plt.legend()
    plt.grid(axis='y', alpha=0.2)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.7, f'{v:.1f}%', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(plots / 'validation_scores.png', dpi=180)
    plt.close()

    with open(args.out / 'validation_scores.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['held_out_video', 'error_vs_zero_percent', 'mse', 'zero_mse'])
        for row in fold_scores:
            w.writerow(row)
        w.writerow(['aggregate', agg, '', ''])

    (args.out / 'summary.txt').write_text(
        'comma.ai Calibration Challenge — Local Validation\n'
        f'Aggregate leave-one-video-out score: {agg:.2f}%\n\n' +
        '\n'.join([f'Video {i}: {pct:.2f}%' for i, pct, _, _ in fold_scores]) +
        '\n\nThis is a local cross-validation result, not an official comma.ai leaderboard score.\n'
    )
    print(f'Results written to {args.out.resolve()}')
    print(f'Aggregate LOVO score: {agg:.2f}%')


if __name__ == '__main__':
    main()
