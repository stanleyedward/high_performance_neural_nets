void relu_cpu(int M, int N, float *d_input, float *d_output){
    int total_elements = M * N;
    for (int i = 0; i < total_elements; i++){
        d_output[i] = (d_input[i] > 0.f) ? d_input[i] : 0.f;
    }
}