void tiled_matrix_multiply_cpu(uint BLOCK_SIZE, uint M, uint N, uint K, float *A, float *B, float *C){
  // Tiling for matrices:
  // A: M x K, B: K x N, C: M x N
  for (unsigned int rowTile = 0; rowTile < M / BLOCK_SIZE; ++rowTile){
    for (unsigned int colTile = 0; colTile < N / BLOCK_SIZE; ++colTile){
      for (unsigned int iTile = 0; iTile < K / BLOCK_SIZE; ++iTile){
        for (unsigned int row = rowTile * BLOCK_SIZE; row < (rowTile + 1) * BLOCK_SIZE; ++row){
          for (unsigned int col = colTile * BLOCK_SIZE; col < (colTile + 1) * BLOCK_SIZE; ++col){
            float sum = 0.0f;
            for (unsigned int i = iTile * BLOCK_SIZE; i < (iTile + 1) * BLOCK_SIZE; ++i){
              sum += A[row * K + i] * B[i * N + col];
            }
            if(iTile == 0)
              C[row * N + col] = sum;
            else
              C[row * N + col] += sum;
          }
        }
      }
    }
  }
}


void matrix_multiply_cpu(uint M, uint N, uint K, float *A, float *B, float *C, float *bias) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int l = 0; l < K; l++) {
                sum += A[i * K + l] * B[l * N + j];
            }
            C[i * N + j] = sum + bias[j];
        }
    }
}