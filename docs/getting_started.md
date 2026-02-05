# Getting Started with Kalbee

This guide will help you integrate `kalbee` into your project.

## Core Concepts

All filters in `kalbee` inherit from `BaseFilter` and share a common interface:

1. **Initialization (`__init__`)**: Setup state ($x$), covariance ($P$), and model matrices.
2. **Prediction (`predict`)**: Propagate the state forward in time. Accepts `dt` (time step).
3. **Update (`update`)**: Correct the state using a new measurement.

## Filter Types

### Kalman Filter (KF)
Best for linear systems with Gaussian noise.
- **Model**: $x_{k} = F x_{k-1} + w_k$
- **Measurement**: $z_k = H x_k + v_k$

### Extended Kalman Filter (EKF)
Best for non-linear systems.
- **Model**: $x_{k} = f(x_{k-1}, dt) + w_k$
- **Measurement**: $z_k = h(x_k) + v_k$
- **Jacobians**: Requires $F$ (Jacobian of $f$) and $H$ (Jacobian of $h$).

### Alpha-Beta-Gamma
Simpler, fixed-gain filter for tracking position, velocity, and acceleration. Good for low-compute environments or simpler dynamics.

## Usage Patterns

You can pass matrices directly or use the `modes` in `AutoFilter`.

```python
# dynamic Jacobian example for EKF
def f_nonlinear(x, dt):
    return # ... some non-linear function
    
ekf.predict(dt=0.1, f=f_nonlinear)
```
