void softmax_cpu(int rows, int cols, float *input, float *output) {
    for (int row = 0; row < rows; row++) {
        float *row_input = input + row * cols;
        float *row_output = output + row * cols;

        // Find the maximum value in the current row to avoid numerical instability
        float maxval = row_input[0];
        for (int i = 1; i < cols; i++) {
            if (row_input[i] > maxval) {
                maxval = row_input[i];
            }
        }

        // Compute the sum of exponentials
        float sum_exp = 0.0f;
        for (int i = 0; i < cols; i++) {
            float exp_val = expf(row_input[i] - maxval);
            row_output[i] = exp_val; // Temporarily store exp values
            sum_exp += exp_val;
        }

        // Normalize by the sum of exponentials
        for (int i = 0; i < cols; i++) {
            row_output[i] /= sum_exp;
        }
    }
}