# TODOs & Roadmap

## 🚀 Enhancements
- [ ] **Unscented Kalman Filter (UKF)**: Implement UKF to handle highly non-linear systems without Jacobians.
- [ ] **Particle Filter**: Add support for particle filtering for non-Gaussian distributions.
- [ ] **Smoothing**: Implement RTS (Rauch-Tung-Striebel) smoothing for post-processing.
- [ ] **Adaptive Filtering**: Implement adaptive noise estimation (estimating Q and R on the fly).

## 🛠 Engineering & Quality
- [ ] **Type Hints**: Improve type hinting coverage, especially for `kwargs` and specific matrix shapes (using `nptyping` or similar if appropriate).
- [ ] **Test Coverage**: Increase test coverage to 90%+. Add more edge cases for singular matrices.
- [ ] **Benchmarks**: Add a benchmarking suite to compare performance against other libraries or raw NumPy implementations.
- [ ] **Vectorization**: Optimize batch processing for filtering multiple objects simultaneously.

## 📚 Documentation
- [ ] **API Reference**: Generate automatic API docs using Sphinx or MkDocs.
- [ ] **Tutorials**: Add Jupyter notebooks demonstrating real-world tracking examples (e.g., radar tracking, robot localization).
