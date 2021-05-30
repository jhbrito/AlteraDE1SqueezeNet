#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <math.h>
#include "AOCLUtils/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "squeezenet_params.h"


#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;
using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
cl_command_queue queue;
cl_program program = NULL;

//host buffer
float *h_result_classifier = (float *)malloc((1000) * sizeof(float)); 
char class_label[201];

cl_kernel conv3x3, conv1x1, maxpool, avgpool;

cl_mem d_sample, d_conv1_weight, d_conv1_bias, d_result_conv;
cl_mem d_result_pool1,d_result_block1_squeeze, d_result_block1_expand;
cl_mem d_fire1_squeeze_weight, d_fire1_squeeze_bias, d_fire1_expand1x1_weight, \
       d_fire1_expand1x1_bias, d_fire1_expand3x3_weight,d_fire1_expand3x3_bias;
cl_mem d_fire2_squeeze_weight, d_fire2_squeeze_bias, d_fire2_expand1x1_weight, \
       d_fire2_expand1x1_bias, d_fire2_expand3x3_weight,d_fire2_expand3x3_bias;
cl_mem d_result_block2_squeeze, d_result_block2_expand, d_result_pool2;
cl_mem d_fire3_squeeze_weight, d_fire3_squeeze_bias, d_fire3_expand1x1_weight, \
       d_fire3_expand1x1_bias, d_fire3_expand3x3_weight,d_fire3_expand3x3_bias;
cl_mem d_fire4_squeeze_weight, d_fire4_squeeze_bias, d_fire4_expand1x1_weight, \
       d_fire4_expand1x1_bias, d_fire4_expand3x3_weight,d_fire4_expand3x3_bias;
cl_mem d_result_pool3, d_result_block3_squeeze1, d_result_block3_expand1, d_result_block3_squeeze2;
cl_mem d_fire5_squeeze_weight, d_fire5_squeeze_bias, d_fire5_expand1x1_weight, \
       d_fire5_expand1x1_bias, d_fire5_expand3x3_weight,d_fire5_expand3x3_bias;
cl_mem d_fire6_squeeze_weight, d_fire6_squeeze_bias, d_fire6_expand1x1_weight, \
       d_fire6_expand1x1_bias, d_fire6_expand3x3_weight,d_fire6_expand3x3_bias;
cl_mem d_fire7_squeeze_weight, d_fire7_squeeze_bias, d_fire7_expand1x1_weight, \
       d_fire7_expand1x1_bias, d_fire7_expand3x3_weight,d_fire7_expand3x3_bias;
cl_mem d_fire8_squeeze_weight, d_fire8_squeeze_bias, d_fire8_expand1x1_weight, \
       d_fire8_expand1x1_bias, d_fire8_expand3x3_weight,d_fire8_expand3x3_bias;
cl_mem d_result_block3_expand2, d_result_classifier_conv, d_classifier_conv_weight, d_classifier_conv_bias,d_result_classifier;

void getLabel(unsigned int class_index);
void cleanup();

// Entry point.
int main() {
  // Initialize OpenCL.
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return 1;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return 1;
  }

  // Query the available OpenCL device.
  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);
  for(unsigned int i = 0; i < num_devices; ++i) {
    printf("  %s\n", getDeviceName(device[i]).c_str());
  }

  // Create the context.
  context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = getBoardBinaryFile("squeezenet", device[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  queue = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  // Kernel.
  const char *kernel1 = "conv2d3x3";
  conv3x3 = clCreateKernel(program, kernel1, &status);
  checkError(status, "Failed to create kernel conv2d3x3");

  const char *kernel2 = "maxpool2d";
  maxpool = clCreateKernel(program, kernel2, &status);
  checkError(status, "Failed to create kernel maxpool2d");

  const char *kernel3 = "conv2d1x1";
  conv1x1 = clCreateKernel(program, kernel3, &status);
  checkError(status, "Failed to create kernel conv2d1x1");

  const char *kernel4 = "avgpool2d";
  avgpool = clCreateKernel(program, kernel4, &status);
  checkError(status, "Failed to create kernel avgpool");
  
  //conv1 params
  d_conv1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(conv1_weight), conv1_weight, &status);
  checkError(status, "Failed to create buffer d_conv1_weight");
  d_conv1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(conv1_bias), conv1_bias, &status);
  checkError(status, "Failed to create buffer d_conv1_bias");

  //conv1 result
  d_result_conv = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                       sizeof(float) * (1 * 64 * 111 * 111), NULL, &status);
  checkError(status, "Failed to create buffer d_result_conv");
  d_result_pool1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                       sizeof(float) * (1 * 64 * 55 * 55), NULL, &status);
  checkError(status, "Failed to create buffer d_result_pool1");

  //fire1
  d_result_block1_squeeze = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                       sizeof(float) * (1 * 16* 55 * 55), NULL, &status);
  checkError(status, "Failed to create buffer d_result_block1_squeeze");
  d_result_block1_expand = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                       sizeof(float) * (1 * 128 * 55 * 55), NULL, &status);
  checkError(status, "Failed to create buffer d_result_block1_expand");
  d_result_pool2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                       sizeof(float) * (1 * 128 * 27 * 27), NULL, &status);
  checkError(status, "Failed to create buffer d_result_pool2");

  d_fire1_squeeze_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire1_squeeze_weight), fire1_squeeze_weight, &status);
  checkError(status, "Failed to create buffer d_fire1_squeeze_weight");
  d_fire1_squeeze_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire1_squeeze_bias), fire1_squeeze_bias, &status);
  checkError(status, "Failed to create buffer d_fire1_squeeze_bias");
  d_fire1_expand1x1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire1_expand1x1_weight), fire1_expand1x1_weight, &status);
  checkError(status, "Failed to create buffer d_fire1_expand1x1_weight");
  d_fire1_expand1x1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire1_expand1x1_bias), fire1_expand1x1_bias, &status);
  checkError(status, "Failed to create buffer d_fire1_expand1x1_bias");
  d_fire1_expand3x3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire1_expand3x3_weight), fire1_expand3x3_weight, &status);
  checkError(status, "Failed to create buffer d_fire1_expand3x3_weight");
  d_fire1_expand3x3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire1_expand3x3_bias), fire1_expand3x3_bias, &status);
  checkError(status, "Failed to create buffer d_fire1_expand3x3_bias");

  //fire2
  d_fire2_squeeze_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire2_squeeze_weight), fire2_squeeze_weight, &status);
  checkError(status, "Failed to create buffer d_fire2_squeeze_weight");
  d_fire2_squeeze_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire2_squeeze_bias), fire2_squeeze_bias, &status);
  checkError(status, "Failed to create buffer d_fire2_squeeze_bias");
  d_fire2_expand1x1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire2_expand1x1_weight), fire2_expand1x1_weight, &status);
  checkError(status, "Failed to create buffer d_fire2_expand1x1_weight");
  d_fire2_expand1x1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire2_expand1x1_bias), fire2_expand1x1_bias, &status);
  checkError(status, "Failed to create buffer d_fire2_expand1x1_bias");
  d_fire2_expand3x3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire2_expand3x3_weight), fire2_expand3x3_weight, &status);
  checkError(status, "Failed to create buffer d_fire2_expand3x3_weight");
  d_fire2_expand3x3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire2_expand3x3_bias), fire2_expand3x3_bias, &status);
  checkError(status, "Failed to create buffer d_fire2_expand3x3_bias");

  //block2
  d_result_block2_squeeze = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                       sizeof(float) * (1 * 32 * 27 * 27), NULL, &status);
  checkError(status, "Failed to create buffer d_result_block2_squeeze");
  d_result_block2_expand = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                       sizeof(float) * (1 * 256 * 27 *27), NULL, &status);
  checkError(status, "Failed to create buffer d_result_block2_expand");
  d_result_pool3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                       sizeof(float) * (1 * 256 * 13 * 13), NULL, &status);
  checkError(status, "Failed to create buffer d_result_pool3");

  //fire3
  d_fire3_squeeze_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire3_squeeze_weight), fire3_squeeze_weight, &status);
  checkError(status, "Failed to create buffer d_fire3_squeeze_weight");
  d_fire3_squeeze_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire3_squeeze_bias), fire3_squeeze_bias, &status);
  checkError(status, "Failed to create buffer d_fire3_squeeze_bias");
  d_fire3_expand1x1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire3_expand1x1_weight), fire3_expand1x1_weight, &status);
  checkError(status, "Failed to create buffer d_fire3_expand1x1_weight");
  d_fire3_expand1x1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire3_expand1x1_bias), fire3_expand1x1_bias, &status);
  checkError(status, "Failed to create buffer d_fire3_expand1x1_bias");
  d_fire3_expand3x3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire3_expand3x3_weight), fire3_expand3x3_weight, &status);
  checkError(status, "Failed to create buffer d_fire3_expand3x3_weight");
  d_fire3_expand3x3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire3_expand3x3_bias), fire3_expand3x3_bias, &status);
  checkError(status, "Failed to create buffer d_fire3_expand3x3_bias");

  //fire4
  d_fire4_squeeze_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire4_squeeze_weight), fire4_squeeze_weight, &status);
  checkError(status, "Failed to create buffer d_fire4_squeeze_weight");
  d_fire4_squeeze_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire4_squeeze_bias), fire4_squeeze_bias, &status);
  checkError(status, "Failed to create buffer d_fire4_squeeze_bias");
  d_fire4_expand1x1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire4_expand1x1_weight), fire4_expand1x1_weight, &status);
  checkError(status, "Failed to create buffer d_fire4_expand1x1_weight");
  d_fire4_expand1x1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire4_expand1x1_bias), fire4_expand1x1_bias, &status);
  checkError(status, "Failed to create buffer d_fire4_expand1x1_bias");
  d_fire4_expand3x3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire4_expand3x3_weight), fire4_expand3x3_weight, &status);
  checkError(status, "Failed to create buffer d_fire4_expand3x3_weight");
  d_fire4_expand3x3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire4_expand3x3_bias), fire4_expand3x3_bias, &status);
  checkError(status, "Failed to create buffer d_fire4_expand3x3_bias");

  d_result_block3_squeeze1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                       sizeof(float) * (1 * 48 * 13 * 13), NULL, &status);
  checkError(status, "Failed to create buffer d_result_block3_squeeze1");
  d_result_block3_expand1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                       sizeof(float) * (1 * 384 * 13 * 13), NULL, &status);
  checkError(status, "Failed to create buffer d_result_block3_expand1");
  d_result_block3_squeeze2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                       sizeof(float) * (1 * 64 * 13 * 13), NULL, &status);
  checkError(status, "Failed to create buffer d_result_block3_squeeze2");
  d_result_block3_expand2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                       sizeof(float) * (1 * 512 * 13 * 13), NULL, &status);
  checkError(status, "Failed to create buffer d_result_block3_expand2");
  
  //fire5
  d_fire5_squeeze_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire5_squeeze_weight), fire5_squeeze_weight, &status);
  checkError(status, "Failed to create buffer d_fire5_squeeze_weight");
  d_fire5_squeeze_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire5_squeeze_bias), fire5_squeeze_bias, &status);
  checkError(status, "Failed to create buffer d_fire5_squeeze_bias");
  d_fire5_expand1x1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire5_expand1x1_weight), fire5_expand1x1_weight, &status);
  checkError(status, "Failed to create buffer d_fire5_expand1x1_weight");
  d_fire5_expand1x1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire5_expand1x1_bias), fire5_expand1x1_bias, &status);
  checkError(status, "Failed to create buffer d_fire5_expand1x1_bias");
  d_fire5_expand3x3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire5_expand3x3_weight), fire5_expand3x3_weight, &status);
  checkError(status, "Failed to create buffer d_fire5_expand3x3_weight");
  d_fire5_expand3x3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire5_expand3x3_bias), fire5_expand3x3_bias, &status);
  checkError(status, "Failed to create buffer d_fire5_expand3x3_bias");

  //fire6
  d_fire6_squeeze_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire6_squeeze_weight), fire6_squeeze_weight, &status);
  checkError(status, "Failed to create buffer d_fire6_squeeze_weight");
  d_fire6_squeeze_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire6_squeeze_bias), fire6_squeeze_bias, &status);
  checkError(status, "Failed to create buffer d_fire6_squeeze_bias");
  d_fire6_expand1x1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire6_expand1x1_weight), fire6_expand1x1_weight, &status);
  checkError(status, "Failed to create buffer d_fire6_expand1x1_weight");
  d_fire6_expand1x1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire6_expand1x1_bias), fire6_expand1x1_bias, &status);
  checkError(status, "Failed to create buffer d_fire6_expand1x1_bias");
  d_fire6_expand3x3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire6_expand3x3_weight), fire6_expand3x3_weight, &status);
  checkError(status, "Failed to create buffer d_fire6_expand3x3_weight");
  d_fire6_expand3x3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire6_expand3x3_bias), fire6_expand3x3_bias, &status);
  checkError(status, "Failed to create buffer d_fire6_expand3x3_bias");

  //fire7
  d_fire7_squeeze_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire7_squeeze_weight), fire7_squeeze_weight, &status);
  checkError(status, "Failed to create buffer d_fire7_squeeze_weight");
  d_fire7_squeeze_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire7_squeeze_bias), fire7_squeeze_bias, &status);
  checkError(status, "Failed to create buffer d_fire7_squeeze_bias");
  d_fire7_expand1x1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire7_expand1x1_weight), fire7_expand1x1_weight, &status);
  checkError(status, "Failed to create buffer d_fire7_expand1x1_weight");
  d_fire7_expand1x1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire7_expand1x1_bias), fire7_expand1x1_bias, &status);
  checkError(status, "Failed to create buffer d_fire7_expand1x1_bias");
  d_fire7_expand3x3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire7_expand3x3_weight), fire7_expand3x3_weight, &status);
  checkError(status, "Failed to create buffer d_fire7_expand3x3_weight");
  d_fire7_expand3x3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire7_expand3x3_bias), fire7_expand3x3_bias, &status);
  checkError(status, "Failed to create buffer d_fire7_expand3x3_bias");

  //fire8
  d_fire8_squeeze_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire8_squeeze_weight), fire8_squeeze_weight, &status);
  checkError(status, "Failed to create buffer d_fire8_squeeze_weight");
  d_fire8_squeeze_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire8_squeeze_bias), fire8_squeeze_bias, &status);
  checkError(status, "Failed to create buffer d_fire8_squeeze_bias");
  d_fire8_expand1x1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire8_expand1x1_weight), fire8_expand1x1_weight, &status);
  checkError(status, "Failed to create buffer d_fire8_expand1x1_weight");
  d_fire8_expand1x1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire8_expand1x1_bias), fire8_expand1x1_bias, &status);
  checkError(status, "Failed to create buffer d_fire8_expand1x1_bias");
  d_fire8_expand3x3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire8_expand3x3_weight), fire8_expand3x3_weight, &status);
  checkError(status, "Failed to create buffer d_fire8_expand3x3_weight");
  d_fire8_expand3x3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(fire8_expand3x3_bias), fire8_expand3x3_bias, &status);
  checkError(status, "Failed to create buffer d_fire8_expand3x3_bias");

  //classifier
  d_classifier_conv_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(classifier_conv_weight), classifier_conv_weight, &status);
  checkError(status, "Failed to create buffer classifier_conv_weight");
  d_classifier_conv_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(classifier_conv_bias), classifier_conv_bias, &status);
  checkError(status, "Failed to create buffer d_classifier_conv_bias");

  d_result_classifier_conv = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                       sizeof(float) * (1 * 1000 * 13 * 13), NULL, &status);
  checkError(status, "Failed to create buffer d_result_classifier_conv");
  d_result_classifier = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                       sizeof(float) * 1000, NULL, &status);
  checkError(status, "Failed to create buffer d_result_classifier");
  
  d_sample = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                       sizeof(float) * (1 * 3 * 224 * 224), NULL, &status);
  checkError(status, "Failed to create buffer d_sample");
  

  printf("\r\nSqueezeNet on FPGA start:\r\n");
  printf("kernel version 1.3\r\n");
  double total = 0.0;
  double total_conv = 0.0;
  double total_readimage = 0.0;
  double total_imageproc = 0.0;
  double total_block1 = 0.0;
  double total_block2 = 0.0;
  double total_block3 = 0.0;
  double total_classificator = 0.0;
  int tp[1000] = { 0 };;
  int fp[1000] = { 0 };;
  int fn[1000] = { 0 };;

  double time_conv[50000] = { 0 };;
  double time_readimage[50000] = { 0 };;
  double time_imageproc[50000] = { 0 };;
  double time_block1[50000] = { 0 };;
  double time_block2[50000] = { 0 };;
  double time_block3[50000] = { 0 };;
  double time_classificator[50000] = { 0 };;

	int input_size1 = 224;
	int new_width, new_height;
	Mat img_in;
	Mat img_resize;
	int new_size = 256;
	float p;
	float media[3] = { 0.485, 0.456, 0.406 };
	float desviop[3] = { 0.229, 0.224, 0.225 };
	float my_array[150528] = { 0 };;;
	

	// new add variables
int dataset[50000] = { 0 };;
int total_img = 50000;
int real;
int matrizconfusion[1000][1000] = { 0 };;
//float accuracy;
int predicted;
int correct_label;
int v = 0;

	// Copiar valores para o Array
char* filepath = new char[100];

FILE *file;
file = fopen("ILSVRC2012_validation_ground_truth_fixed.txt", "r");
int num;
int d=0;
while(fscanf(file,"%d",&num)>0){
dataset[d] = num;
d++;}
fclose(file);

double start_time = getCurrentTimestamp();
for (int a = 30001; a <= total_img; a++){ // for all images{
sprintf(filepath, "/storage/ILSVRC2012_img_val/ILSVRC2012_val_%08d.JPEG", a);
img_in = imread(filepath);
double end_time = getCurrentTimestamp();
total_readimage += (end_time - start_time);
printf("Ler imagem: %f s \n", (end_time - start_time));
total += (end_time - start_time);
time_readimage[v] = (end_time - start_time);
start_time = end_time;
		// cols(colunas) -> largura(width)  --  rows(linhas) -> altura(height)
		if (img_in.cols > img_in.rows) {
			new_width = new_size * img_in.cols / img_in.rows;
			new_height = new_size;
		}
		else {
			new_width = new_size;
			new_height = new_size * img_in.rows / img_in.cols;
		}

		// RESIZE da imagem.
		resize(img_in, img_resize, Size(new_width, new_height));

		// CROP da imagem.
		const int cropSize = 224;
		const int offsetW = (img_resize.cols - cropSize) / 2;
		const int offsetH = (img_resize.rows - cropSize) / 2;
		const Rect roi(offsetW, offsetH, cropSize, cropSize);
		img_resize = img_resize(roi).clone();

		// Retirar o número de canais da imagem.

		int cn = img_resize.channels();
		cvtColor(img_resize, img_resize, COLOR_BGR2RGB);
		// Retirar o valor dos pixéis e salvar
		for (int c = 0; c < 3; c++) {
			for (int h = 0; h < img_resize.rows; h++) {
				for (int w = 0; w < img_resize.cols; w++) {
					Vec3b a = img_resize.at<Vec3b>(h, w);
					p = (float)a.val[c];
					p = p / 255;
					p = (p - media[c]) / desviop[c];
					my_array[(c)*input_size1 * input_size1 + h * input_size1 + w] = p;
				}
			}
		}

  clEnqueueWriteBuffer(queue, d_sample, CL_TRUE, 0, sizeof(float) * (1 * 3 * 224 * 224), my_array, 0, NULL, NULL);
  end_time = getCurrentTimestamp();
  total_imageproc += (end_time - start_time);
  printf("Tempo de Pre-Processamento: %f s \n", (end_time - start_time));
  total += (end_time - start_time);
  time_imageproc[v] = (end_time - start_time);
  start_time = end_time;
  /**************************************************************/
  /*                          conv1                             */
  /**************************************************************/


  unsigned int input_channel, input_size, pad, stride, start_channel, output_size;
  
  input_channel = 3;
  input_size = 224;
  pad = 0;
  stride = 2;
  start_channel = 0;
  output_size = 111;

  status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
  status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
  status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
  status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
  status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_sample);
  status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_conv1_weight);
  status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_conv1_bias);
  status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_conv);
  checkError(status, "Setting conv1: conv3x3 arguments");

  size_t global_f[2] = {64,111};
  status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing conv1: conv3x3");

  input_size = 111;
  output_size = 55;
  status |= clSetKernelArg(maxpool, 0, sizeof(int), &(input_size));
  status |= clSetKernelArg(maxpool, 1, sizeof(int), &(output_size));
  status |= clSetKernelArg(maxpool, 2, sizeof(cl_mem), &d_result_conv);
  status |= clSetKernelArg(maxpool, 3, sizeof(cl_mem), &d_result_pool1);
  checkError(status, "Setting maxpool1 arguments");

  size_t global = 64;
  status = clEnqueueNDRangeKernel(queue, maxpool, 1, NULL, &global, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing maxpool1");

  status = clFinish(queue);
  checkError(status, "Wait for maxpool1 finish");

  end_time = getCurrentTimestamp();
  total_conv += (end_time - start_time);
   printf("Tempo de Conv1: %f s \n", (end_time - start_time));
  time_conv[v] = (end_time - start_time);
  total += (end_time - start_time);
  start_time = end_time;
  /**************************************************************/
  /*                         block1                             */
  /**************************************************************/
  //fire1
  input_channel = 64 / 4;
  input_size = 55;

  status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_pool1);
  status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire1_squeeze_weight);
  status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire1_squeeze_bias);
  status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block1_squeeze);
  checkError(status, "Setting fire1_squeeze arguments");

  global_f[0] = 16; global_f[1] = 55;
  status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire1_squeeze");

  input_channel = 16 / 4;
  input_size = 55;

  status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block1_squeeze);
  status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire1_expand1x1_weight);
  status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire1_expand1x1_bias);
  status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block1_expand);
  checkError(status, "Setting fire1_expand1x1 arguments");

  global_f[0] = 64; global_f[1] = 55;
  status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire1_expand1x1");

  status = clFinish(queue);
  checkError(status, "Wait for block1 finish");

  input_channel = 16;
  input_size = 55;
  pad = 1;
  stride = 1;
  start_channel = 64;
  output_size = 55;

  status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
  status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
  status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
  status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
  status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_result_block1_squeeze);
  status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_fire1_expand3x3_weight);
  status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_fire1_expand3x3_bias);
  status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_block1_expand);
  checkError(status, "Setting fire1_expand3x3 arguments");

  global_f[0] = 64; global_f[1] = 55;
  status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire1_expand3x3");

  //fire2
  input_channel = 128 / 4;
  input_size = 55;

  status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block1_expand);
  status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire2_squeeze_weight);
  status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire2_squeeze_bias);
  status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block1_squeeze);
  checkError(status, "Setting fire2_squeeze arguments");

  global_f[0] = 16; global_f[1] = 55;
  status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire2_squeeze");

  input_channel = 16 / 4;
  input_size = 55;

  status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block1_squeeze);
  status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire2_expand1x1_weight);
  status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire2_expand1x1_bias);
  status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block1_expand);
  checkError(status, "Setting fire2_expand1x1 arguments");

  global_f[0] = 64; global_f[1] = 55;
  status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire2_expand1x1");

  input_channel = 16;
  input_size = 55;
  pad = 1;
  stride = 1;
  start_channel = 64;
  output_size = 55;

  status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
  status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
  status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
  status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
  status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_result_block1_squeeze);
  status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_fire2_expand3x3_weight);
  status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_fire2_expand3x3_bias);
  status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_block1_expand);
  checkError(status, "Setting fire2_expand3x3 arguments");

  global_f[0] = 64; global_f[1] = 55;
  status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire2_expand3x3");
  
  input_size = 55;
  output_size = 27;
  status |= clSetKernelArg(maxpool, 0, sizeof(int), &(input_size));
  status |= clSetKernelArg(maxpool, 1, sizeof(int), &(output_size));
  status |= clSetKernelArg(maxpool, 2, sizeof(cl_mem), &d_result_block1_expand);
  status |= clSetKernelArg(maxpool, 3, sizeof(cl_mem), &d_result_pool2);
  checkError(status, "Setting maxpool2 arguments");

  global = 128;
  status = clEnqueueNDRangeKernel(queue, maxpool, 1, NULL, &global, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing block1: maxpool");

  status = clFinish(queue);
  checkError(status, "Wait for block1 finish");

  end_time = getCurrentTimestamp();
  total_block1 += (end_time - start_time);
  printf("Tempo de Block1: %f s \n", (end_time - start_time));
  total += (end_time - start_time);
  time_block1[v] = (end_time - start_time);
  start_time = end_time;

  /**************************************************************/
  /*                         block2                             */
  /**************************************************************/
  //fire3
  input_channel = 128 / 4;
  input_size = 27;

  status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_pool2);
  status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire3_squeeze_weight);
  status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire3_squeeze_bias);
  status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block2_squeeze);
  checkError(status, "Setting fire3_squeeze arguments");

  global_f[0] = 32; global_f[1] = 27;
  status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire3_squeeze");

  input_channel = 32 / 4;
  input_size = 27;

  status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block2_squeeze);
  status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire3_expand1x1_weight);
  status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire3_expand1x1_bias);
  status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block2_expand);
  checkError(status, "Setting fire3_expand1x1 arguments");

  global_f[0] = 128; global_f[1] = 27;
  status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire3_expand1x1");
  
  input_channel = 32;
  input_size = 27;
  pad = 1;
  stride = 1;
  start_channel = 128;
  output_size = 27;

  status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
  status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
  status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
  status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
  status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_result_block2_squeeze);
  status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_fire3_expand3x3_weight);
  status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_fire3_expand3x3_bias);
  status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_block2_expand);
  checkError(status, "Setting fire3_expand3x3 arguments");

  global_f[0] = 128; global_f[1] = 27;
  status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire3_expand3x3");

  status = clFinish(queue);
  checkError(status, "Wait for fire3 finish");

  //fire4
  input_channel = 256 / 4;
  input_size = 27;

  status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block2_expand);
  status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire4_squeeze_weight);
  status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire4_squeeze_bias);
  status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block2_squeeze);
  checkError(status, "Setting fire4_squeeze arguments");

  global_f[0] = 32; global_f[1] = 27;
  status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire4_squeeze");

  input_channel = 32 / 4;
  input_size = 27;

  status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block2_squeeze);
  status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire4_expand1x1_weight);
  status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire4_expand1x1_bias);
  status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block2_expand);
  checkError(status, "Setting fire4_expand1x1 arguments");

  global_f[0] = 128; global_f[1] = 27;
  status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire4_expand1x1");

  input_channel = 32;
  input_size = 27;
  pad = 1;
  stride = 1;
  start_channel = 128;
  output_size = 27;

  status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
  status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
  status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
  status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
  status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_result_block2_squeeze);
  status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_fire4_expand3x3_weight);
  status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_fire4_expand3x3_bias);
  status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_block2_expand);
  checkError(status, "Setting fire4_expand3x3 arguments");

  global_f[0] = 128; global_f[1] = 27;
  status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire4_expand3x3");

  status = clFinish(queue);
  checkError(status, "Wait for fire4 finish");

  input_size = 27;
  output_size = 13;
  status |= clSetKernelArg(maxpool, 0, sizeof(int), &(input_size));
  status |= clSetKernelArg(maxpool, 1, sizeof(int), &(output_size));
  status |= clSetKernelArg(maxpool, 2, sizeof(cl_mem), &d_result_block2_expand);
  status |= clSetKernelArg(maxpool, 3, sizeof(cl_mem), &d_result_pool3);
  checkError(status, "Setting block2: maxpool arguments");

  global = 256;
  status = clEnqueueNDRangeKernel(queue, maxpool, 1, NULL, &global, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing block2: maxpool");

  status = clFinish(queue);
  checkError(status, "Wait for block2 finish");

  end_time = getCurrentTimestamp();
  total += (end_time - start_time);
  printf("Tempo de Block2: %f s \n", (end_time - start_time));
  total_block2 += (end_time - start_time);
  time_block2[v] = (end_time - start_time);
  start_time = end_time;

  /**************************************************************/
  /*                         block3                             */
  /**************************************************************/
  //fire5
  input_channel = 256 / 4;
  input_size = 13;

  status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_pool3);
  status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire5_squeeze_weight);
  status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire5_squeeze_bias);
  status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block3_squeeze1);
  checkError(status, "Setting fire5_squeeze arguments");

  global_f[0] = 48; global_f[1] = 13;
  status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire5_squeeze");

  input_channel = 48 / 4;
  input_size = 13;

  status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block3_squeeze1);
  status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire5_expand1x1_weight);
  status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire5_expand1x1_bias);
  status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block3_expand1);
  checkError(status, "Setting fire5_expand1x1 arguments");

  global_f[0] = 192; global_f[1] = 13;
  status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire5_expand1x1");

  input_channel = 48;
  input_size = 13;
  pad = 1;
  stride = 1;
  start_channel = 192;
  output_size = 13;

  status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
  status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
  status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
  status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
  status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_result_block3_squeeze1);
  status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_fire5_expand3x3_weight);
  status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_fire5_expand3x3_bias);
  status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_block3_expand1);
  checkError(status, "Setting fire5_expand3x3 arguments");

  global_f[0] = 192; global_f[1] = 13;
  status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire5_expand3x3");

  status = clFinish(queue);
  checkError(status, "Wait for fire5 finish");

  //fire6
  input_channel = 384 / 4;
  input_size = 13;

  status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block3_expand1);
  status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire6_squeeze_weight);
  status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire6_squeeze_bias);
  status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block3_squeeze1);
  checkError(status, "Setting fire6_squeeze arguments");

  global_f[0] = 48; global_f[1] = 13;
  status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire6_squeeze");

  input_channel = 48 / 4;
  input_size = 13;

  status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block3_squeeze1);
  status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire6_expand1x1_weight);
  status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire6_expand1x1_bias);
  status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block3_expand1);
  checkError(status, "Setting fire6_expand1x1 arguments");

  global_f[0] = 192; global_f[1] = 13;
  status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire6_expand1x1");

  input_channel = 48;
  input_size = 13;
  pad = 1;
  stride = 1;
  start_channel = 192;
  output_size = 13;

  status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
  status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
  status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
  status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
  status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_result_block3_squeeze1);
  status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_fire6_expand3x3_weight);
  status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_fire6_expand3x3_bias);
  status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_block3_expand1);
  checkError(status, "Setting fire6_expand3x3 arguments");

  global_f[0] = 192; global_f[1] = 13;
  status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire6_expand3x3");

  status = clFinish(queue);
  checkError(status, "Wait for fire6 finish");

  //fire7
  input_channel = 384 / 4;
  input_size = 13;

  status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block3_expand1);
  status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire7_squeeze_weight);
  status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire7_squeeze_bias);
  status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block3_squeeze2);
  checkError(status, "Setting fire7_squeeze arguments");

  global_f[0] = 64; global_f[1] = 13;
  status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire7_squeeze");

  input_channel = 64 / 4;
  input_size = 13;

  status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block3_squeeze2);
  status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire7_expand1x1_weight);
  status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire7_expand1x1_bias);
  status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block3_expand2);
  checkError(status, "Setting fire7_expand1x1 arguments");

  global_f[0] = 256; global_f[1] = 13;
  status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire7_expand1x1");

  input_channel = 64;
  input_size = 13;
  pad = 1;
  stride = 1;
  start_channel = 256;
  output_size = 13;

  status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
  status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
  status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
  status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
  status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_result_block3_squeeze2);
  status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_fire7_expand3x3_weight);
  status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_fire7_expand3x3_bias);
  status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_block3_expand2);
  checkError(status, "Setting fire7_expand3x3 arguments");

  global_f[0] = 256; global_f[1] = 13;
  status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire7_expand3x3");

  status = clFinish(queue);
  checkError(status, "Wait for fire7 finish");

  //fire8
  input_channel = 512 / 4;
  input_size = 13;

  status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block3_expand2);
  status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire8_squeeze_weight);
  status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire8_squeeze_bias);
  status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block3_squeeze2);
  checkError(status, "Setting fire8_squeeze arguments");

  global_f[0] = 64; global_f[1] = 13;
  status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire8_squeeze");

  input_channel = 64 / 4;
  input_size = 13;

  status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block3_squeeze2);
  status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_fire8_expand1x1_weight);
  status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_fire8_expand1x1_bias);
  status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_block3_expand2);
  checkError(status, "Setting fire8_expand1x1 arguments");

  global_f[0] = 256; global_f[1] = 13;
  status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire8_expand1x1");

  input_channel = 64;
  input_size = 13;
  pad = 1;
  stride = 1;
  start_channel = 256;
  output_size = 13;

  status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
  status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
  status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
  status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
  status |= clSetKernelArg(conv3x3, 6, sizeof(cl_mem), &d_result_block3_squeeze2);
  status |= clSetKernelArg(conv3x3, 7, sizeof(cl_mem), &d_fire8_expand3x3_weight);
  status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_fire8_expand3x3_bias);
  status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_result_block3_expand2);
  checkError(status, "Setting fire8_expand3x3 arguments");

  global_f[0] = 256; global_f[1] = 13;
  status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing fire8_expand3x3");

  status = clFinish(queue);
  checkError(status, "Wait for fire8 finish");

  end_time = getCurrentTimestamp();
  total += (end_time - start_time);
  printf("Tempo de Block3: %f s \n", (end_time - start_time));
  total_block3 += (end_time - start_time);
  time_block3[v] = (end_time - start_time);
  start_time = end_time;

  /**************************************************************/
  /*                       classifier                           */
  /**************************************************************/
  input_channel = 512 / 4;
  input_size = 13;

  status |= clSetKernelArg(conv1x1, 0, sizeof(int), &(input_channel));
  status |= clSetKernelArg(conv1x1, 1, sizeof(int), &(input_size));
  status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_result_block3_expand2);
  status |= clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &d_classifier_conv_weight);
  status |= clSetKernelArg(conv1x1, 4, sizeof(cl_mem), &d_classifier_conv_bias);
  status |= clSetKernelArg(conv1x1, 5, sizeof(cl_mem), &d_result_classifier_conv);
  checkError(status, "Setting classifier_conv arguments");

  global_f[0] = 1000; global_f[1] = 13;
  status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, global_f, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing classifier_conv");

  status |= clSetKernelArg(avgpool, 0, sizeof(cl_mem), &d_result_classifier_conv);
  status |= clSetKernelArg(avgpool, 1, sizeof(cl_mem), &d_result_classifier);
  checkError(status, "Setting avgpool arguments");

  global = 1000;
  status = clEnqueueNDRangeKernel(queue, avgpool, 1, NULL, &global, NULL, 0, NULL, NULL);
  checkError(status, "Enqueueing avgpool");

  status = clFinish(queue);
  checkError(status, "Wait for classifier finish");
  end_time = getCurrentTimestamp();
  status = clEnqueueReadBuffer(queue, d_result_classifier, CL_TRUE, 0, sizeof(float) * 1000, h_result_classifier, 0, NULL, NULL );
  float tmp = 0.0f;
  int class_index = -1;
  // Este j é quem dá a classe
  for(int j = 0; j < 1000; j++)
  {
    if(h_result_classifier[j] > tmp)
    {
  	  tmp = h_result_classifier[j];
      class_index = j;

    }
  }

  real = dataset[a-1];
  printf("classe index %d\n", class_index);
  printf("classe real %d\n", real);

if (real == class_index) {
      correct_label = correct_label + 1;
      tp[real] = tp[real]+1;
	  matrizconfusion[real][class_index]++;
	}else{
fn[real] = fn[real] + 1;
fp[class_index] = fp[class_index] + 1;
matrizconfusion[real][class_index]++;
}


  total += (end_time - start_time);
  total_classificator += (end_time - start_time);
  printf("Tempo de Classificar: %f s \n", (end_time - start_time));
  time_classificator[v] = (end_time - start_time);
  getLabel(class_index);
  v++;
  //printf("%d\n", v);
    printf("total: %f s \n", total);

} // TERMINA AQUI O CICLO

printf("Copia de valores Temporais");
FILE *ValorTempo;
ValorTempo = fopen("ValorTempo.txt", "w");
fprintf(ValorTempo,"Total de tempo de Leitura de imagem %f \n", total_readimage);
fprintf(ValorTempo,"Total de tempo de Processar Imagem %f \n", total_imageproc);
fprintf(ValorTempo,"Total de tempo de Convolucao %f \n", total_conv);
fprintf(ValorTempo,"Total de tempo do bloco 1 %f \n", total_block1);
fprintf(ValorTempo,"Total de tempo do bloco 2 %f \n", total_block2);
fprintf(ValorTempo,"Total de tempo do bloco 3 %f \n", total_block3);
fprintf(ValorTempo,"Total de tempo de classificar a imagem %f \n", total_classificator);
fclose(ValorTempo);

// METRICAS
/*
float precision[1000];
float fscore[1000];
float recall[1000];
int classes = 0;
int total_classes = 1000;
*/
FILE *Metricas;
Metricas = fopen("Metricas.txt", "w");
fprintf(Metricas,"Total correct label = : %d", correct_label); 
fclose(Metricas);
/*
for(classes = 0; classes < total_classes; classes++)
{
	if((tp[classes] + fp[classes])==0){precision[classes]=0;}
	else{precision[classes]= (tp[classes]) / (tp[classes] + fp[classes]);}
	
	if((tp[classes] + fn[classes])==0){recall[classes]=0;}
	else{recall[classes]= (tp[classes]) / (tp[classes] + fn[classes]);}
	
	if((precision[classes] + recall[classes])==0){fscore[classes]=0;}
	else{fscore[classes]= (2 * precision[classes] * recall[classes]) / (precision[classes] + recall[classes]);}
	
	fprintf(Metricas,"Classe: %d precisao: %f, recall: %f, fscore: %f \n", classes, precision[classes], recall[classes], fscore[classes]);
}
fclose(Metricas);
*/
int total_classes = 1000;
ofstream MatrizConfusao;
MatrizConfusao.open("MatrizConfusao.txt");
 for(int r = 0; r < total_classes; r++){
    for(int g = 0; g < total_classes; g++){
	MatrizConfusao << matrizconfusion[r][g] << " ";
	if(g == (total_classes - 1)){
		MatrizConfusao << endl;
	}
	}
 }
 MatrizConfusao.close();
 //printf("Guardar Matriz. \n");
 //accuracy = (correct_label)/total_img;
 //printf("\nAccuracy: %f \n", accuracy);

/*
// Calculo de Media
double avg_time_conv = 0.0;
double avg_time_readimage = 0.0;
double avg_time_imageproc = 0.0;
double avg_time_block1 = 0.0;
double avg_time_block2 = 0.0;
double avg_time_block3 = 0.0;
double avg_time_classificator = 0.0;

avg_time_readimage = (total_readimage) / total_img;
avg_time_imageproc = (total_imageproc) / total_img;
avg_time_conv = (total_conv) / total_img;
avg_time_block1 = (total_block1) / total_img;
avg_time_block2 = (total_block2) / total_img;
avg_time_block3 = (total_block3) / total_img;
avg_time_classificator = (total_classificator) / total_img;
printf("Media verificada. \n");

// Calculo de Desvio Padrao
double SD_readimage = 0.0;
double SD_imageproc = 0.0;
double SD_conv = 0.0;
double SD_block1 = 0.0;
double SD_block2 = 0.0;
double SD_block3 = 0.0;
double SD_classificator = 0.0;


for (int s = 0; s < total_img; s++){
SD_readimage = SD_readimage + pow(time_readimage[s] - (avg_time_readimage), 2); 
SD_imageproc = SD_imageproc + pow(time_imageproc[s] - (avg_time_imageproc), 2); 
SD_conv = SD_conv + pow(time_conv[s] - (avg_time_conv), 2); 
SD_block1 = SD_block1 + pow(time_block1[s] - (avg_time_block1), 2); 
SD_block2 = SD_block2 + pow(time_block2[s] - (avg_time_block2), 2); 	
SD_block3 = SD_block3 +  pow(time_block3[s] - (avg_time_block3), 2);  
SD_classificator = SD_classificator + pow(time_classificator[s] - (avg_time_classificator), 2); 
}

printf("Desvio padrao 1 verificado. \n");
SD_readimage = sqrt(SD_readimage/total_img);
SD_imageproc = sqrt(SD_imageproc/total_img);
SD_conv = sqrt(SD_conv/total_img);
SD_block1 = sqrt(SD_block1/total_img);
SD_block2 = sqrt(SD_block2/total_img);
SD_block3 = sqrt(SD_block3/total_img);
SD_classificator = sqrt(SD_classificator/total_img);
printf("Desvio padrao 2 verificado. \n");

//Criar Ficheiro e guardar valores de Media e Desvio Padrao

ofstream outputFile;
outputFile.open("Media_DesvioPadrao.txt");
outputFile <<"Tempo Medio de Abrir Ficheiro: " << avg_time_readimage << " s " << " Desvio Padrao de: " << SD_readimage << endl;
outputFile <<"Tempo Medio de Pre Processar Imagem: " << avg_time_imageproc << " s "  << " Desvio Padrao de: " << SD_imageproc << endl;
outputFile <<"Tempo Medio de Convolucao: " << avg_time_conv << " s " << " Desvio Padrao de: " << SD_conv << endl;
outputFile <<"Tempo Medio de Block1: " << avg_time_block1 << " s " << " Desvio Padrao de: " << SD_block1 << endl;
outputFile <<"Tempo Medio de Block2: " << avg_time_block2 << " s " << " Desvio Padrao de: " << SD_block2 << endl;
outputFile <<"Tempo Medio de Block3: " << avg_time_block3 << " s " << " Desvio Padrao de: " << SD_block3 << endl;
outputFile <<"Tempo Medio de Classificar: " << avg_time_classificator << " Desvio Padrao de: " << SD_classificator << endl;
outputFile.close();
printf("Guardar verificado. \n");
*/
  return 0;
}

void getLabel(unsigned int class_index)
{
  int i;
  
  FILE *fp;

  fp = fopen("synset_words.txt", "r");
  for(i = 0; i < class_index + 1; i++)
  {
    fgets(class_label, sizeof(class_label), fp);
  }
  fclose(fp);
}

void cleanup()
{
  clReleaseMemObject(d_sample);
  clReleaseMemObject(d_conv1_weight);
  clReleaseMemObject(d_conv1_bias);
  clReleaseMemObject(d_result_conv);

  clReleaseMemObject(d_result_pool1);
  clReleaseMemObject(d_result_block1_squeeze);
  clReleaseMemObject(d_result_block1_expand);

  clReleaseMemObject(d_fire1_squeeze_weight);
  clReleaseMemObject(d_fire1_squeeze_bias);
  clReleaseMemObject(d_fire1_expand1x1_weight);
  clReleaseMemObject(d_fire1_expand1x1_bias);
  clReleaseMemObject(d_fire1_expand3x3_weight);
  clReleaseMemObject(d_fire1_expand3x3_bias);

  clReleaseMemObject(d_fire2_squeeze_weight);
  clReleaseMemObject(d_fire2_squeeze_bias);
  clReleaseMemObject(d_fire2_expand1x1_weight);
  clReleaseMemObject(d_fire2_expand1x1_bias);
  clReleaseMemObject(d_fire2_expand3x3_weight);
  clReleaseMemObject(d_fire2_expand3x3_bias);

  clReleaseMemObject(d_result_block2_squeeze);
  clReleaseMemObject(d_result_block2_expand);
  clReleaseMemObject(d_result_pool2);

  clReleaseMemObject(d_fire3_squeeze_weight);
  clReleaseMemObject(d_fire3_squeeze_bias);
  clReleaseMemObject(d_fire3_expand1x1_weight);
  clReleaseMemObject(d_fire3_expand1x1_bias);
  clReleaseMemObject(d_fire3_expand3x3_weight);
  clReleaseMemObject(d_fire3_expand3x3_bias);

  clReleaseMemObject(d_fire4_squeeze_weight);
  clReleaseMemObject(d_fire4_squeeze_bias);
  clReleaseMemObject(d_fire4_expand1x1_weight);
  clReleaseMemObject(d_fire4_expand1x1_bias);
  clReleaseMemObject(d_fire4_expand3x3_weight);
  clReleaseMemObject(d_fire4_expand3x3_bias);

  clReleaseMemObject(d_result_pool3);
  clReleaseMemObject(d_result_block3_squeeze1);
  clReleaseMemObject(d_result_block3_expand1);
  clReleaseMemObject(d_result_block3_squeeze2);

  clReleaseMemObject(d_fire5_squeeze_weight);
  clReleaseMemObject(d_fire5_squeeze_bias);
  clReleaseMemObject(d_fire5_expand1x1_weight);
  clReleaseMemObject(d_fire5_expand1x1_bias);
  clReleaseMemObject(d_fire5_expand3x3_weight);
  clReleaseMemObject(d_fire5_expand3x3_bias);

  clReleaseMemObject(d_fire6_squeeze_weight);
  clReleaseMemObject(d_fire6_squeeze_bias);
  clReleaseMemObject(d_fire6_expand1x1_weight);
  clReleaseMemObject(d_fire6_expand1x1_bias);
  clReleaseMemObject(d_fire6_expand3x3_weight);
  clReleaseMemObject(d_fire6_expand3x3_bias);

  clReleaseMemObject(d_fire7_squeeze_weight);
  clReleaseMemObject(d_fire7_squeeze_bias);
  clReleaseMemObject(d_fire7_expand1x1_weight);
  clReleaseMemObject(d_fire7_expand1x1_bias);
  clReleaseMemObject(d_fire7_expand3x3_weight);
  clReleaseMemObject(d_fire7_expand3x3_bias);

  clReleaseMemObject(d_fire8_squeeze_weight);
  clReleaseMemObject(d_fire8_squeeze_bias);
  clReleaseMemObject(d_fire8_expand1x1_weight);
  clReleaseMemObject(d_fire8_expand1x1_bias);
  clReleaseMemObject(d_fire8_expand3x3_weight);
  clReleaseMemObject(d_fire8_expand3x3_bias);

  clReleaseMemObject(d_result_block3_expand2);
  clReleaseMemObject(d_result_classifier_conv);

  clReleaseMemObject(d_classifier_conv_weight);
  clReleaseMemObject(d_classifier_conv_bias);
  clReleaseMemObject(d_result_classifier);

  clReleaseKernel(conv3x3);
  clReleaseKernel(conv1x1);
  clReleaseKernel(maxpool);
  clReleaseKernel(avgpool);
	
  clReleaseCommandQueue(queue);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);

  free(h_result_classifier);
}

