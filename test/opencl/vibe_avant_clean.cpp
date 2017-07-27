    #include <CL/cl.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>

/*
const char* programSource =
"__kernel \n"
"void vecadd( __global char* img, __global int* background) \n"
"{                              \n"
"   int x = get_global_id(0); \n"
"   img[x] = img[x] + 10; \n"
"}";
*/
char* readSource(char* kernelPath);

int main()
{
    const int elements = 2048*2048;
    size_t datasize = sizeof(int) * elements;


    int *A = (int*) malloc(elements * sizeof(int));
    int *B = (int*) malloc(elements * sizeof(int));
    int *C = (int*) malloc(elements * sizeof(int));
    int i = 0;
    for(i = 0; i < elements; ++i) {
        A[i] = i;
        B[i] = i;
    }

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
    std::cout << shore_mask.elemSize() << std::endl;

    cl_int ret_code = CL_SUCCESS;
    size_t img_datasize = shore_mask.cols * shore_mask.rows;

    int nb_elements = img_datasize * 20;
    size_t background_size = nb_elements * 4;
    int* h_background = (int*)malloc(background_size);
    for(int i = 0; i < nb_elements; ++i) h_background[i] = 0;

    cl_mem shore_mask_buff = clCreateBuffer(context, CL_MEM_READ_WRITE, img_datasize,  shore_mask.data, &ret_code);
    cl_mem background_buff = clCreateBuffer(context, CL_MEM_READ_WRITE, background_size, h_background, &ret_code);

    //-------------------------------------------------------------------------------------

    // allocate 2 input and one output buffer for the three vectors in the vector addition
    // cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
    // cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);

    // cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);

    // write data from the niput arrays to the bufers
    // status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_TRUE, 0, datasize, A, 0, NULL, NULL);
    // status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_TRUE, 0, datasize, B, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmdQueue, shore_mask_buff, CL_TRUE, 0, img_datasize, shore_mask.data, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmdQueue, background_buff, CL_TRUE, 0, background_size, h_background, 0, NULL, NULL);

    // create a program with source code
    // cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);
    // if(status != 0) std::cout << "Error create prog" << std::endl;
    // build the program for the device
    // status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);


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

    // set the kernel arguments
    // status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    // status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    // status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &shore_mask_buff);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &background_buff);
    // status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImage);

    // define an index space of work-items for execution
    // a work-group size is not required but can be used
    size_t indexSpaceSize = 432*288;
    // size_t workGroupSize[1] = {8};
    // Execute the kernel
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, &indexSpaceSize, NULL, 0, NULL, NULL);

    // read the device output buffer to the host outupt array
    // status = clEnqueueReadBuffer(cmdQueue, bufC, CL_TRUE, 0, datasize, C, 0, NULL, NULL);
    // cv::imwrite("out.jpg", shore_mask);
    status = clEnqueueReadBuffer(cmdQueue, shore_mask_buff, CL_TRUE, 0, shore_mask.cols * shore_mask.rows , (void*)shore_mask.data, 0, NULL, NULL);
    status = clEnqueueReadBuffer(cmdQueue, background_buff, CL_TRUE, 0, shore_mask.cols * shore_mask.rows * 20 * 4, (void*)h_background, 0, NULL, NULL);

    // std::cout << shore_mask << std::endl;
    cv::imwrite("out2.jpg", shore_mask);
    for(int i = 0; i < nb_elements; i++)
    {
        printf("%d ", h_background[i]);
    }
    printf("\n");

    /*
    int endPrint = 10;
    printf("Resultats : \n");
    printf("A : \n");
    for(i = 0; i < endPrint; ++i) printf("%d ", A[i]); printf("\n");
    printf("B : \n");
    for(i = 0; i < endPrint; ++i) printf("%d ", B[i]); printf("\n");
    printf("C : \n");
    for(i = 0; i < endPrint; ++i) printf("%d ", C[i]); printf("\n");
    */

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
