import numpy as np
import random

def generate_w1a_like(
    n_samples=1000000,
    n_features=300,
    avg_active=10,
    seed=42,
    noise=0.0,
    output_file="synthetic_data.txt"
):
    np.random.seed(seed)
    random.seed(seed)

    # True weight vector (what SGD should recover)
    x_true = np.random.randn(n_features)

    with open(output_file, "w") as f:
        for _ in range(n_samples):
            k = np.random.poisson(avg_active)
            k = max(1, min(k, n_features))

            features = sorted(random.sample(range(n_features), k))

            row = np.zeros(n_features)
            row[features] = 1

            y = np.sign(row @ x_true)
            if y == 0:
                y = 1

            if random.random() < noise:
                y *= -1

            feature_str = " ".join(f"{idx+1}:1" for idx in features)
            f.write(f"{ '+1' if y == 1 else '-1' } {feature_str}\n")

    print(f"Saved to {output_file}")

if __name__ == "__main__":
    generate_w1a_like()