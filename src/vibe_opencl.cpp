
#include "common.hpp"
#include <CL/cl.h>
#include <stdio.h>
typedef unsigned long long int iterator;

char* readSource(char* kernelPath);

// local implementation for ViBe segmentation algorithm
struct ViBe_impl : ViBe {
    ViBe_impl(size_t N, size_t R, size_t nMin, size_t nSigma);
    virtual ~ViBe_impl();
    virtual void initialize(const cv::Mat& oInitFrame);
    virtual void apply(const cv::Mat& oCurrFrame, cv::Mat& oOutputMask);
    const size_t m_N; //< internal ViBe parameter; number of samples to keep in each pixel model
    const size_t m_R; //< internal ViBe parameter; maximum color distance allowed between RGB samples for matching
    const size_t m_nMin; //< internal ViBe parameter; required number of matches for background classification
    const size_t m_nSigma; //< internal ViBe parameter; model update rate

    // OpenCL
    cl_int ret_code;
    cl_int status;
    cl_platform_id *platforms;
    cl_device_id *devices;
    cl_uint numDevices;
    cl_context context;
    cl_command_queue cmdQueue;

    size_t img_size;
    size_t background_size;

    unsigned char* h_background;

    cl_mem d_image;
    cl_mem d_output;
    cl_mem d_background;
    cl_mem d_random;

    cl_program program;
    cl_kernel kernel;

//    unsigned int* h_debug;
//    cl_mem debug_buff;
};

std::shared_ptr<ViBe> ViBe::createInstance(size_t N, size_t R, size_t nMin, size_t nSigma) {
    return std::shared_ptr<ViBe>(new ViBe_impl(N,R,nMin,nSigma));
}

ViBe_impl::ViBe_impl(size_t N, size_t R, size_t nMin, size_t nSigma) :
    m_N(N),
    m_R(R),
    m_nMin(nMin),
    m_nSigma(nSigma) {

    cl_uint numPlatforms = 0;

    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    platforms = (cl_platform_id*) malloc(numPlatforms * sizeof(cl_platform_id));
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);

    numDevices = 0;
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
    devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
    status =clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
    cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
    img_size = 320 * 240;
    background_size = img_size * 3 * 20;
    h_background = (unsigned char*)new unsigned int[background_size * sizeof(unsigned char)]();

//    h_debug = (unsigned int*)new unsigned int[img_size];
//    for(iterator i = 0; i < img_size; ++i)
//        h_debug[i] = 0;

    d_image = clCreateBuffer(context, CL_MEM_READ_ONLY, img_size * 3,  NULL, &ret_code);
    d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, img_size,  NULL, &ret_code);
    d_background = clCreateBuffer(context, CL_MEM_READ_WRITE, background_size * sizeof(unsigned char), NULL, &ret_code); //maybe use char istead of int to save memory
    d_random = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * sizeof(int), NULL, &ret_code);
//    debug_buff = clCreateBuffer(context, CL_MEM_READ_WRITE, 4 * (img_size / 3), NULL, &ret_code);

}

ViBe_impl::~ViBe_impl() {
    delete h_background;
    clReleaseDevice(*devices);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseContext(context);
}


void ViBe_impl::initialize(const cv::Mat& oInitFrame) {
    CV_Assert(!oInitFrame.empty() && oInitFrame.isContinuous() && oInitFrame.type()==CV_8UC3);

    for(int i = 1; i < oInitFrame.rows-1; i++)
    {
        for(int j = 1; j < oInitFrame.cols-1; ++j)
        {

            cv::Vec3b neighbours[] = {oInitFrame.at<cv::Vec3b>(i-1,j-1), oInitFrame.at<cv::Vec3b>(i-1,j), oInitFrame.at<cv::Vec3b>(i-1,j+1), oInitFrame.at<cv::Vec3b>(i,j-1),
                                  oInitFrame.at<cv::Vec3b>(i,j+1), oInitFrame.at<cv::Vec3b>(i+1,j-1), oInitFrame.at<cv::Vec3b>(i+1,j), oInitFrame.at<cv::Vec3b>(i+1,j+1)};
            for(int k = 0; k < 20; ++k)
            {
                h_background[i * 3 * 20 * oInitFrame.cols + (j * 3 * 20 )+ (k * 3)] = (unsigned char) neighbours[rand() % 8][0];
                h_background[i * 3 * 20 * oInitFrame.cols + (j * 3 * 20 )+ (k * 3) + 1] = (unsigned char) neighbours[rand() % 8][1];
                h_background[i * 3 * 20 * oInitFrame.cols + (j * 3 * 20 )+ (k * 3) + 2] = (unsigned char) neighbours[rand() % 8][2];
            }
        }
    }

    status = clEnqueueWriteBuffer(cmdQueue, d_background, CL_TRUE, 0, background_size * sizeof(unsigned char), h_background, 0, NULL, NULL);

    // read kernel
    char* progSource = readSource(KERNEL_FILE);
    size_t progSize = strlen(progSource);
    program = clCreateProgramWithSource(context, 1, (const char**)&progSource,
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

    // create kernel
    kernel = clCreateKernel(program, "vecadd", &status);
    if(status != 0) std::cout << "Error create kernel" << std::endl;

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_image);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_background);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_random);
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_output);
//    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &debug_buff);
}




void ViBe_impl::apply(const cv::Mat& oCurrFrame, cv::Mat& oOutputMask) {
    CV_Assert(!oCurrFrame.empty() && oCurrFrame.isContinuous() && oCurrFrame.type()==CV_8UC3);
    oOutputMask.create(oCurrFrame.size(),CV_8UC1); //TODO output is binary, but always stored in a byte (so output values are either '0' or '255')
    size_t frame_size = oCurrFrame.cols * oCurrFrame.rows;

    int randoms[2] = {rand() % 16, rand() % 20};
    status = clEnqueueWriteBuffer(cmdQueue, d_image, CL_TRUE, 0, frame_size * 3, oCurrFrame.data, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmdQueue, d_random, CL_TRUE, 0, 2*sizeof(int), randoms, 0, NULL, NULL);
//    status = clEnqueueWriteBuffer(cmdQueue, debug_buff, CL_TRUE, 0, img_size*4, h_debug, 0, NULL, NULL);

    size_t indexSpaceSize = frame_size;
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, &indexSpaceSize, NULL, 0, NULL, NULL);
    status = clEnqueueReadBuffer(cmdQueue, d_output, CL_TRUE, 0, frame_size, oOutputMask.data, 0, NULL, NULL);
//    status = clEnqueueReadBuffer(cmdQueue, debug_buff, CL_TRUE, 0, frame_size * 4, h_debug, 0, NULL, NULL);

   // filtre median
   cv::medianBlur(oOutputMask, oOutputMask, 9);
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
