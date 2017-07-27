#include <CL/cl.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

const char* programSource =
"__kernel \n"
"void vecadd( image2d_t __read_only img, image2d_t __write_only out) \n"
"{                              \n"
"   int x = get_global_id(0); \n"
"   int y = get_global_id(1); \n"
"   int4 color;      \n"
"   const sampler_t samplerA = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;; \n"
"   color = read_imagei(img, samplerA, (int2)(x, y)); \n"
"   int4 color2;    \n"
"   color2 = (int4)(10, 0, 0, 0);"
"   write_imagei(out, (int2)(x, y), color2);    \n"
"}";


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
    cv::Mat shore_mask = cv::imread("lkz.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    std::cout << shore_mask.elemSize() << std::endl;

    cl_int ret_code = CL_SUCCESS;
    // size_t img_datasize = shore_mask.cols * shore_mask.rows * 8;

    // cl_mem shore_mask_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, img_datasize, (void*)shore_mask.data, &ret_code);


    cl_image_format format = { CL_INTENSITY, CL_UNORM_INT8 };


    cl_mem inputImage = clCreateImage2D (context,
        CL_MEM_READ_WRITE, &format,
        shore_mask.cols, shore_mask.rows, 0, (void*)shore_mask.data,
        &ret_code);
    cl_mem outputImage = clCreateImage2D (context,
        CL_MEM_READ_WRITE, &format,
        shore_mask.cols, shore_mask.rows, 0, NULL,
        &ret_code);
    if(ret_code != 0)   std::cout << "ERROR" << std::endl;

    //-------------------------------------------------------------------------------------

    // allocate 2 input and one output buffer for the three vectors in the vector addition
    // cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
    // cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);

    // cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);

    // write data from the niput arrays to the bufers
    // status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_TRUE, 0, datasize, A, 0, NULL, NULL);
    // status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_TRUE, 0, datasize, B, 0, NULL, NULL);
    const size_t offset[3] = {0,0,0};
    const size_t region[3] = {64, 64, 1};
    status = clEnqueueWriteImage(cmdQueue, inputImage, CL_TRUE, offset, region, 0, 0, shore_mask.data, 0, NULL, NULL);
    // status = clEnqueueWriteImage(cmdQueue, outputImage, CL_TRUE, offset, region, 0, 0, shore_mask.data, 0, NULL, NULL);
    if(status != 0) std::cout << "Error enqueue img" << std::endl;

    // status = clEnqueueWriteBuffer(cmdQueue, inputImage, CL_TRUE, 0, img_datasize, shore_mask.data, 0, NULL, NULL);

    // create a program with source code
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);
    if(status != 0) std::cout << "Error create prog" << std::endl;
    // build the program for the device
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
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImage);

    // define an index space of work-items for execution
    // a work-group size is not required but can be used
    size_t indexSpaceSize[2] = {64,64};
    size_t workGroupSize[2] = {8, 8};
    // Execute the kernel
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, indexSpaceSize, workGroupSize, 0, NULL, NULL);

    int* hOutput = (int*)malloc(sizeof(int)*256);
    // read the device output buffer to the host outupt array
    // status = clEnqueueReadBuffer(cmdQueue, bufC, CL_TRUE, 0, datasize, C, 0, NULL, NULL);
    cv::imwrite("out.jpg", shore_mask);
    status = clEnqueueReadImage(cmdQueue, outputImage, CL_TRUE, offset, region, 0, 0, shore_mask.data, 0, NULL, NULL);
    // status = clEnqueueReadBuffer(cmdQueue, inputImage, CL_TRUE, 0, shore_mask.cols * shore_mask.rows * 8, (void*)shore_mask.data, 0, NULL, NULL);
    std::cout << shore_mask << std::endl;
    cv::imwrite("out2.jpg", shore_mask);

    return 0;
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
