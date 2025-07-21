# Performance Comparison

This project compares the performance of different implementations of a mean filter over a 2D grid, with increasing levels of optimization. Each element in the grid is updated with the average of its row and column neighbors (excluding itself), using various techniques such as blocking, SIMD intrinsics, and OpenMP for parallelism.

## Implementations Overview

### 🔵 Reference (Basic)
- This is the **slowest** implementation, especially for large grid sizes.
- It lacks any form of optimization — **no blocking**, **no SIMD**, and **no multithreading**.
- Execution time grows **exponentially** with the size of the grid.
- **Not scalable**, and becomes infeasible for large inputs.
- Used as a **baseline** to compare the effectiveness of other optimized versions.

---

### 🟢 Blocked
- **Faster than the reference**, noticeable from `4096-8` and beyond.
- Introduces **blocking**, which improves **cache locality** by working on smaller chunks.
- This version scales better and utilizes cache more efficiently than the reference.
- However, it is still **single-threaded**.

---

### 🟩 SIMD
- **Faster than the blocked version**, particularly on large grids.
- Combines **blocking + SIMD intrinsics** to perform row/column summation in parallel using SSE (processing 4 floats at a time).
- Significant reduction in computation time due to **data-level parallelism**.
- Still runs on a **single core** — no thread-level parallelism yet.

---

### 🔴 SIMD + OpenMP
- The **fastest and most scalable** version across all configurations.
- Combines **blocking + SIMD + multithreading (OpenMP)**.
- Leverages both:
  - **Data-level parallelism** (SIMD)
  - **Thread-level parallelism** (OpenMP)
- This maximizes **CPU resource utilization**.
- Execution time grows **very slowly**, even for very large grid sizes.
- Fully parallel and **scales excellently** on multi-core systems.

