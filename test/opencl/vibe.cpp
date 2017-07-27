#include <CL/cl.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>

char* readSource(char* kernelPath);

int main()
{
    cl_int status;

    // retrieve nb of platforms
    cl_uint numPlatforms = 0;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    // allocate enough space for each platform
    cl_platform_id *platforms = NULL;
    platforms = (cl_platform_id*) malloc(numPlatforms * sizeof(cl_platform_id));
    // fill in the platforms
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    // retrieve nb of devices
    cl_uint numDevices = 0;
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
    // allocate enough space for each device
    cl_device_id *devices;
    devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
    // fill in the devices
    status =clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
    // create a context that include all devices
    cl_context context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
    // only create a command-queue for the first device
    cl_command_queue cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);

    //-----------------------------------------------------------------------------------
    cv::Mat shore_mask = cv::imread("img.png", CV_LOAD_IMAGE_GRAYSCALE);

    size_t img_datasize = shore_mask.cols * shore_mask.rows;

    int nb_elements = img_datasize * 20;
    size_t background_size = nb_elements * 4;
    int* h_background = (int*)malloc(background_size);
    for(int i = 0; i < nb_elements; ++i) h_background[i] = 0;
    cl_int ret_code = CL_SUCCESS;
    cl_mem shore_mask_buff = clCreateBuffer(context, CL_MEM_READ_WRITE, img_datasize,  shore_mask.data, &ret_code);
    cl_mem background_buff = clCreateBuffer(context, CL_MEM_READ_WRITE, background_size, h_background, &ret_code);

    //-------------------------------------------------------------------------------------

    status = clEnqueueWriteBuffer(cmdQueue, shore_mask_buff, CL_TRUE, 0, img_datasize, shore_mask.data, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmdQueue, background_buff, CL_TRUE, 0, background_size, h_background, 0, NULL, NULL);

    // read kernel
    char* progSource = readSource("kernel.cl");
    size_t progSize = strlen(progSource);
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&progSource,
                            &progSize, &status);

    status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

    if(status != 0) {
        std::cout << "Error build prog " << status << std::endl;
        size_t log_size;
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
    }

    // create the vector addition kernel
    cl_kernel kernel = clCreateKernel(program, "vecadd", &status);
    if(status != 0) std::cout << "Error create kernel" << std::endl;

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &shore_mask_buff);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &background_buff);

    size_t indexSpaceSize = 432*288;
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, &indexSpaceSize, NULL, 0, NULL, NULL);

    status = clEnqueueReadBuffer(cmdQueue, shore_mask_buff, CL_TRUE, 0, shore_mask.cols * shore_mask.rows , (void*)shore_mask.data, 0, NULL, NULL);
    status = clEnqueueReadBuffer(cmdQueue, background_buff, CL_TRUE, 0, shore_mask.cols * shore_mask.rows * 20 * 4, (void*)h_background, 0, NULL, NULL);

    // std::cout << shore_mask << std::endl;
    cv::imwrite("out2.jpg", shore_mask);
    for(int i = 0; i < nb_elements; i++)
    {
        printf("%d ", h_background[i]);
    }
    printf("\n");

    // release resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    // clReleaseMemObject(bufA);
    // clReleaseMemObject(bufB);
    // clReleaseMemObject(bufC);
    clReleaseContext(context);


}


// This function reads in a text file and stores it as a char pointer
char* readSource(char* kernelPath) {

   cl_int status;
   FILE *fp;
   char *source;
   long int size;

   printf("Program file is: %s\n", kernelPath);

   fp = fopen(kernelPath, "rb");
   if(!fp) {
      printf("Could not open kernel file\n");
      exit(-1);
   }
   status = fseek(fp, 0, SEEK_END);
   if(status != 0) {
      printf("Error seeking to end of file\n");
      exit(-1);
   }
   size = ftell(fp);
   if(size < 0) {
      printf("Error getting file position\n");
      exit(-1);
   }

   rewind(fp);

   source = (char *)malloc(size + 1);

   int i;
   for (i = 0; i < size+1; i++) {
      source[i]='\0';
   }

   if(source == NULL) {
      printf("Error allocating space for the kernel source\n");
      exit(-1);
   }

   fread(source, 1, size, fp);
   source[size] = '\0';

   return source;
}
