#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// =========================================================
// FUNCTION PROTOTYPE
// =========================================================
void sparse__multiply(
    int rows,
    int cols,
    const double* A,
    const double* x,
    int* out_nnz,
    double* values,
    int* col_indices,
    int* row_ptrs,
    double* y
);

// =========================================================
// TODO: USER IMPLEMENTATION
// =========================================================
/**
 * @brief   Multiplies a sparse matrix A (in dense format) with a dense vector x, producing a dense output vector y.
 *
 * The function should:
 *      1. Scans a row-major matrix A and identifies its non-zero elements.
 *      2. Extracts them into Compressed Sparse Row (CSR) format using caller-provided buffers.
 *      3. Computes the matrix-vector product y = A * x using the extracted CSR data.
 *      4. Writes the result directly into a caller-provided output buffer.
 * @param   rows Number of rows in the matrix A.
 * @param   cols Number of columns in the matrix A.
 * @param   A Pointer to the dense matrix A, stored in row-major order (size rows*cols).
 * @param   x Pointer to the dense vector x (size cols).
 * @param   out_nnz Pointer to an integer where the function will store the number of non-zero entries in A.
 * @param   values Pointer to an array where the function will store the non-zero values of A in CSR format (size at least rows*cols).
 * @param   col_indices Pointer to an array where the function will store the column indices of the non-zero values in A (size at least rows*cols).
 * @param   row_ptrs Pointer to an array where the function will store the row pointers for the CSR format (size at least rows+1).
 * @param   y Pointer to an array where the function will store the resulting vector y (size rows).
 * @note    The function should not perform any dynamic memory allocation; all necessary buffers are provided by the caller.
*/
void sparse_multiply(
    int rows, int cols, const double* A, const double* x,
    int* out_nnz, double* values, int* col_indices, int* row_ptrs,
    double* y
) {
    if(rows <= 0 || cols <= 0 || !A || !x || !out_nnz || !values || !col_indices || !row_ptrs || !y) 
    {
        return;
    }
    
    row_ptrs[0] = 0;
    int buffer_nnz = 0;
    int current_index = 0;

    for(int i = 0; i < rows; ++i)
    {
        int row_nnz = 0;
        for(int j = 0; j < cols; ++j)
        {
            int offset = (i*cols) + j;
            double value = A[offset];
            if(fabs(value) > 1e-12)
            {
                row_nnz++;
                values[current_index] = value;
                col_indices[current_index++] = j;
            }
        }
        buffer_nnz += row_nnz;
        row_ptrs[i + 1] = buffer_nnz;
    }
    *out_nnz = buffer_nnz;

    for(int i = 0; i < rows; ++i)
    {
        double sum = 0.0f;
        for(int idx = row_ptrs[i]; idx < row_ptrs[i + 1]; ++idx)
        {
            sum += values[idx] * x[col_indices[idx]];
        }
        y[i] = sum;
    }
}

// =========================================================
// TEST HARNESS
// =========================================================
int main(void) {
    srand(time(NULL));
    
    const int num_iterations = 100;
    int passed_count = 0;

    for (int iter = 0; iter < num_iterations; ++iter) {
        int rows = rand() % 41 + 5;
        int cols = rand() % 41 + 5;
        double density = 0.05 + (rand() / (double) RAND_MAX) * 0.35;
        
        size_t mat_sz = (size_t) rows * cols;

        double* A = calloc(mat_sz, sizeof(double));
        for (size_t i = 0; i < mat_sz; ++i) {
            if (((double) rand() / RAND_MAX) < density) {
                A[i] = ((double) rand() / RAND_MAX) * 20.0 - 10.0;
            }
        }

        double* values = malloc(mat_sz * sizeof(double));
        int* col_indices = malloc(mat_sz * sizeof(int));
        int* row_ptrs = malloc((rows + 1) * sizeof(int));
        double* x = malloc(cols * sizeof(double));
        double* y_user = malloc(rows * sizeof(double));
        double* y_ref = calloc(rows, sizeof(double));
        int out_nnz = 0;

        for (int i = 0; i < cols; ++i) {
            x[i] = ((double) rand() / RAND_MAX) * 20.0 - 10.0;
        }

        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;
            for (int j = 0; j < cols; ++j) {
                sum += A[i * cols + j] * x[j];
            }
            y_ref[i] = sum;
        }

        sparse_multiply(rows, cols, A, x, &out_nnz, values, col_indices, row_ptrs, y_user);

        double max_err = 0.0;
        int passed = 1;
        for (int i = 0; i < rows; ++i) {
            double diff = fabs(y_user[i] - y_ref[i]);
            double tol = 1e-7 + 1e-7 * fabs(y_ref[i]); // Mixed absolute/relative tolerance
            if (diff > tol) {
                max_err = fmax(max_err, diff);
                passed = 0;
            }
        }

        if (passed) {
            passed_count++;
        }

        printf(
            "Iter %2d [%3dx%3d, density=%.2f, nnz=%4d]: %s (Max error: %.2e)\n",
            iter, rows, cols, density, out_nnz, passed ? "PASS" : "FAIL", max_err
        );

        free(A);
        free(values);
        free(col_indices);
        free(row_ptrs);
        free(x);
        free(y_user);
        free(y_ref);
    }

    printf(
        "\n%s (%d/%d iterations passed)\n",
        passed_count == num_iterations ? "All tests passed!" : "Some tests failed.",
        passed_count, num_iterations
    );
           
    return passed_count == num_iterations ? 0 : 1;
}

