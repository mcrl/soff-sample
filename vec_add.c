#include <stdio.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

void alloc_vec(float **m, int N) {
  *m = (float *) malloc(sizeof(float) * N);
  if (*m == NULL) {
    printf("Failed to allocate memory for vector.\n");
    exit(0);
  }
}

void rand_vec(float *m, int N) {
  for (int i = 0; i < N; i++) { 
    m[i] = (float) rand() / RAND_MAX - 0.5;
  }
}

void check_vec_add(float *A, float *B, float *C, int N) {
  printf("Validating...\n");

  float *C_ans;
  alloc_vec(&C_ans, N);
  for (int i = 0; i < N; ++i) {
    C_ans[i] = A[i] + B[i];
  }

  int is_valid = 1;
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  for (int i = 0; i < N; ++i) {
    float c = C[i];
    float c_ans = C_ans[i];
    if (fabsf(c - c_ans) > eps && (c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps)) {
      ++cnt;
      if (cnt <= thr)
        printf("C[%d] : correct_value = %f, your_value = %f\n", i, c_ans, c);
      if (cnt == thr + 1)
        printf("Too many error, only first %d values are printed.\n", thr);
      is_valid = 0;
    }
  }

  if (is_valid) {
    printf("Result: VALID\n");
  } else {
    printf("Result: INVALID\n");
  }
}

void print_platform_info(cl_platform_id platform) {
  size_t sz;
  char *buf;
  CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &sz));
  buf = (char*)malloc(sz);
  CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sz, buf, NULL));
  printf("Detected OpenCL platform: %s\n", buf);
  free(buf);
}

void print_device_info(cl_device_id device) {
  size_t sz;
  char *buf;
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &sz));
  buf = (char*)malloc(sz);
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, sz, buf, NULL));
  printf("Detected OpenCL device: %s\n", buf);
  free(buf);
}

unsigned char* load_binary(const char *filename, size_t *size) {
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    return NULL;
  }
  fseek(file, 0, SEEK_END);
  *size = ftell(file);
  rewind(file);
  unsigned char *binary = (unsigned char*)malloc(*size);
  fread(binary, 1, *size, file);
  fclose(file);
  return binary;
}

cl_program create_and_build_program_with_binary(cl_context context, cl_device_id device, const char *file_name) {
  cl_int err;
  size_t binary_size;
  unsigned char *binary = load_binary(file_name, &binary_size);
  if (binary == NULL) {
    printf("Failed to open %s\n", file_name);
    exit(EXIT_FAILURE);
  }
  cl_program program = clCreateProgramWithBinary(context, 1, &device, &binary_size, (const unsigned char**)&binary, NULL, &err);
  CHECK_ERROR(err);
  free(binary);
  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
    char *log = (char*)malloc(log_size + 1);
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL));
    log[log_size] = 0;
    printf("Compile error:\n%s\n", log);
    free(log);
  }
  CHECK_ERROR(err);
  return program;
}

int main() {
  srand(time(NULL));

  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;
  cl_mem a_d, b_d, c_d;

  // Get OpenCL platform
  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);
  print_platform_info(platform);

  // Get OpenCL device
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, 
      &device, NULL);
  CHECK_ERROR(err);
  print_device_info(device);

  // Create OpenCL context
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);

  // Create OpenCL command queue
  queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERROR(err);

  // Compile program from "kernel.cl"
  program = create_and_build_program_with_binary(context, device, "kernel.cl.sfb");

  // Extract kernel from compiled program
  kernel = clCreateKernel(program, "vec_add", &err); 
  CHECK_ERROR(err);

  int N = 1024;
  float *A, *B, *C;
  alloc_vec(&A, N);
  alloc_vec(&B, N);
  alloc_vec(&C, N);
  rand_vec(A, N);
  rand_vec(B, N);

  // Create buffers
  a_d = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &err);
  CHECK_ERROR(err);
  b_d = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &err);
  CHECK_ERROR(err);
  c_d = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &err);
  CHECK_ERROR(err);

  // Write to device
  err = clEnqueueWriteBuffer(queue, a_d, CL_TRUE, 0, N * sizeof(float), A, 0, NULL, NULL);
  CHECK_ERROR(err);
  err = clEnqueueWriteBuffer(queue, b_d, CL_TRUE, 0, N * sizeof(float), B, 0, NULL, NULL);
  CHECK_ERROR(err);

  // Setup kernel arguments
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 3, sizeof(int), &N);
  CHECK_ERROR(err);

  // Setup global work size and local work size
  size_t gws[1] = {N}, lws[1] = {256};
  for (int i = 0; i < 1; ++i) {
    // By OpenCL spec, global work size should be MULTIPLE of local work size
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  // Run kernel
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gws, lws, 0, NULL, NULL);
  CHECK_ERROR(err);

  // Read from device
  err = clEnqueueReadBuffer(queue, c_d, CL_TRUE, 0, N * sizeof(float), C, 0, NULL, NULL);
  CHECK_ERROR(err);

  check_vec_add(A, B, C, N);

  clReleaseMemObject(a_d);
  clReleaseMemObject(b_d);
  clReleaseMemObject(c_d);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return 0;
}
