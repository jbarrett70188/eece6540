#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef AOCL
  #include "CL/opencl.h"
  #include "AOCLUtils/aocl_utils.h"
  using namespace aocl_utils;
  void cleanup();
#endif

#define MAX_SOURCE_SIZE (0x100000)
#define DEVICE_NAME_LEN 128
static char dev_name[DEVICE_NAME_LEN];

int main()
{
  cl_uint platformCount;
  cl_platform_id* platforms;
  cl_device_id device_id;
  cl_uint ret_num_devices;
  cl_int ret;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;

  FILE *fp;
  char fileName[] = "./mykernel.cl";
  char *source_str;
  size_t source_size;

  #ifdef AOCL
    // Get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);

    // Get the OpenCL platform.
    platforms[0] = findPlatform("Intel(R) FPGA");

    if(platforms[0] == NULL)
    {
      printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
      return false;
    }

    // Query the available OpenCL device.
    getDevices(platforms[0], CL_DEVICE_TYPE_ALL, &ret_num_devices);
    printf("Platform: %s\n", getPlatformName(platforms[0]).c_str());
    printf("Using one out of %d device(s)\n", ret_num_devices);
    ret = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    printf("device name=  %s\n", getDeviceName(device_id).c_str());

  #else
    #error "unknown OpenCL SDK environment"
  #endif

  /* Create OpenCL context */
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

  /* Create Command Queue */
  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

  #ifdef AOCL
    /* Create Kernel Program from the binary */
    std::string binary_file = getBoardBinaryFile("mykernel", device_id);
    printf("Using AOCX: %s\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), &device_id, 1);
  #else
    #error "unknown OpenCL SDK environment"
  #endif

  /* Build Kernel Program */
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (ret != CL_SUCCESS)
  {
    printf("Failed to build program.\n");
     exit(1);
  }

  /* Create OpenCL Kernel */
  kernel = clCreateKernel(program, "calculatePi", &ret);
  if (ret != CL_SUCCESS)
  {
    printf("Failed to create kernel.\n");
    exit(1);
  }

  /* Create an array to hold the differences */
  size_t globalws[2]={2048, 1};
  size_t localws[2] = {2, 1};
  float *difference_array = (float *)calloc (globalws[0],  sizeof(float));
  float sum = 0.0f;
    
  /* Allocate space for the difference array on the device */
  cl_mem difference_array_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, globalws[0]*sizeof(float), NULL, &ret);

  /* Set the kernel arguments */
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&difference_array_buffer);

  /* Execute the kernel */
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalws, localws, 0, NULL, NULL);

  /* It's important to check the return value. For example, when enqueueNDRangeKernel may fail when Work group size */
  /* does not divide evenly into global work size */
  if (ret != CL_SUCCESS)
  {
    printf("Failed to enqueueNDRangeKernel.\n");
    exit(1);
  }

  /* Copy the output data back to the host */
  clEnqueueReadBuffer(command_queue, difference_array_buffer, CL_TRUE, 0, globalws[0]*sizeof(float), (void *)difference_array, 0, NULL, NULL);

  /* Verify the results */
  for (int i = 0; i < globalws[0]; i++)
  {
    sum += difference_array[i];
  }

  printf ("Pi = %.4f\n", sum*4);

  /* Free resources */
  free(difference_array);

  clReleaseMemObject(difference_array_buffer);
  clReleaseCommandQueue(command_queue);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseContext(context);

  return 0;
}

#ifdef AOCL
  // Altera OpenCL needs this callback function implemented in main.c
  // Free the resources allocated during initialization
  void cleanup()
  {
  }
#endif
