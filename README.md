# kalbee 🐝

<div align="center">
  <img src="docs/kalbee.png" alt="kalbee logo" width="300"/>
</div>

<br>

`kalbee` is a clean, modular Python implementation of Kalman Filters and related estimation algorithms. Designed for simplicity and performance, it provides a standard interface for state estimation in various applications.

## ✨ Features

- **Standard Kalman Filter (KF)**: Linear estimation for dynamic systems.
- **Extended Kalman Filter (EKF)**: Support for non-linear systems using Jacobians.
- **Alpha-Beta-Gamma Filter**: Lightweight filtering for kinematic tracking.
- **AutoFilter**: Factory pattern for easy filter instantiation.
- **Modular Design**: Easy to extend with new filter types.
- **NumPy Integration**: Optimized for numerical computations.
- **Robustness**: Uses Joseph form for numerically stable covariance updates.

## 🚀 Installation

You can install `kalbee` directly from the source:

```bash
git clone https://github.com/MinLee0210/kalbee.git
cd kalbee
pip install .
```

Or using `uv` (recommended):

```bash
uv pip install -e .
```

## 🛠️ Quick Start

### 1. Standard Kalman Filter

```python
import numpy as np
from kalbee.modules.filters import KalmanFilter

# Define matrices
state = np.zeros((2, 1))  # [position, velocity]
cov = np.eye(2)
F = np.array([[1, 1], [0, 1]])  # Constant velocity model (dt=1)
Q = np.eye(2) * 0.01
H = np.array([[1, 0]])          # We measure position
R = np.array([[0.1]])

# Initialize
kf = KalmanFilter(state, cov, F, Q, H, R)

# Predict & Update
kf.predict()
kf.update(np.array([[1.2]]))

print(f"Estimated State:\n{kf.x}")
```

### 2. AutoFilter Factory

Easily switch between filters using the `AutoFilter` factory:

```python
from kalbee.modules.filters import AutoFilter

# Create an EKF
ekf = AutoFilter.from_filter(
    state, cov, Q, R,
    mode='ekf',
    transition_function=my_f,
    measurement_function=my_h
)
```

## 📚 Documentation

Detailed documentation and examples can be found in the `docs/` directory.

- [Getting Started](docs/getting_started.md)
- [Filtering Logic](docs/filtering_logic.md)

## 🧪 Testing

Run these commands to test the library:

```bash
uv run pytest tests/
```

## 📄 License

This project is licensed under the Apache License 2.0.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Check `TODO.md` for ideas.