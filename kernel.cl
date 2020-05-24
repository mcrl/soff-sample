__kernel void vec_add(__global float *A, __global float *B, __global float *C, int N) {
  int i = get_global_id(0);
  if (i >= N) return;
  C[i] = A[i] + B[i];
}
