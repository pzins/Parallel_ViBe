#include <CL/cl.h>
#include <stdio.h>


const char* programSource =
"__kernel \n"
"void vecadd(__global int* A,   \n"
"            __global int* B,   \n"
"            __global int* C)   \n"
"{                              \n"
"   int idx = get_global_id(0); \n"
"   C[idx] = A[idx] + B[idx];   \n"
"}";


void main()
{
    const int elements = 2048;
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


    // allocate 2 input and one output buffer for the three vectors in the vector addition
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);

    // write data from the niput arrays to the bufers
    status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_TRUE, 0, datasize, A, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_TRUE, 0, datasize, B, 0, NULL, NULL);

    // create a program with source code
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);

    // build the pro#include <opencv2/opencv.hpp>
gram for the device
    status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

    // create the vector addition kernel
    cl_kernel kernel = clCreateKernel(program, "vecadd", &status);

    // set the kernel arguments
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    // define an index space of work-items for execution
    // a work-group size is not required but can be used
    size_t indexSpaceSize[1], workGroupSize[1];
    indexSpaceSize[0] = datasize/sizeof(int);
    workGroupSize[0] = 256;

    // Execute the kernel
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, indexSpaceSize, workGroupSize, 0, NULL, NULL);

    // read the device output buffer to the host outupt array
    status = clEnqueueReadBuffer(cmdQueue, bufC, CL_TRUE, 0, datasize, C, 0, NULL, NULL);

    int endPrint = 10;
    printf("Resultats : \n");
    printf("A : \n");
    for(i = 0; i < endPrint; ++i) printf("%d ", A[i]); printf("\n");
    printf("B : \n");
    for(i = 0; i < endPrint; ++i) printf("%d ", B[i]); printf("\n");
    printf("C : \n");
    for(i = 0; i < endPrint; ++i) printf("%d ", C[i]); printf("\n");

    // release resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseContext(context);

}
