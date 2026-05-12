# RV-Sparse: Matrix-Vector Multiplication

## Overview
This repository contains a C implementation of a Sparse Matrix-Vector (SpMV) multiplication routine. The core function, `sparse_multiply`, is designed to be highly memory-efficient by operating under a strict **zero dynamic memory allocation** constraint. 

It dynamically scans a dense row-major matrix, extracts its non-zero elements into the **Compressed Sparse Row (CSR)** format, and computes the matrix-vector product $y = A \times x$, utilizing only caller-provided memory buffers.

## Algorithm and Implementation

The `sparse_multiply` function performs its operations in two logically distinct phases:

### 1. Matrix Compression (Dense to CSR)
The function iterates through the input dense matrix $A$. To account for floating-point arithmetic noise, elements with an absolute value strictly greater than a small tolerance (e.g., `1e-12`) are classified as non-zero. 

For each non-zero element discovered:
* The value is appended to the `values` array.
* Its original column index is appended to the `col_indices` array.
* The `row_ptrs` array is continuously updated to mark the starting and ending indices of non-zero elements for each specific row.

### 2. Matrix-Vector Multiplication (SpMV)
Using the newly populated CSR arrays, the function computes the product vector $y$:
* It loops through each row $i$ from `0` to `rows - 1`.
* It retrieves the start and end indices for the current row from `row_ptrs[i]` and `row_ptrs[i + 1]`.
* It computes the dot product of the row's non-zero values and the corresponding elements in the dense vector $x$.
* The final scalar sum is written directly to the caller-provided array `y[i]`.

## Constraints & Memory Management
Per the challenge requirements, this implementation performs **zero dynamic memory allocation**. 
* No calls to `malloc`, `calloc`, or `realloc` are made within `sparse_multiply`.
* The caller is entirely responsible for allocating and freeing the worst-case sized buffers (`values`, `col_indices`, `row_ptrs`, and `y`).
* The function safely writes exactly `out_nnz` elements to the CSR arrays to prevent buffer overflows.

## How to Build and Run

A test harness (`challenge.c`) is provided to validate the correctness of the implementation across randomized matrix dimensions, sparsity densities, and values.

To compile and execute the test harness, run the following commands in your terminal:

```bash
# Compile the C source file, linking the math library
gcc -lm -o run challenge.c

# Execute the compiled binary
./run

The output will display the results of multiple iterations, comparing the CSR sparse multiplication output against a standard dense matrix-vector multiplication reference to ensure accuracy within acceptable floating-point tolerances.