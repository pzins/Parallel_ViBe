
#include "tp1/common.hpp"
#include <CL/cl.h>
#include <CL/sycl.hpp>
typedef unsigned long long int iterator;
using namespace cl::sycl;

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

    uchar* h_background;
	int h_randoms[2];


    cl_program program;
    cl_kernel kernel;

//    unsigned int* h_debug;
//    cl_mem debug_buff;
    gpu_selector mySelector;
    cl::sycl::context myContext;
	queue myQueue;
//	buffer<uchar, 1> dd_image;
	buffer<uchar, 1> d_background;
	//buffer<uchar, 1> dd_output;
	buffer<int, 1> d_randoms;
};

std::shared_ptr<ViBe> ViBe::createInstance(size_t N, size_t R, size_t nMin, size_t nSigma) {
    return std::shared_ptr<ViBe>(new ViBe_impl(N,R,nMin,nSigma));
}

ViBe_impl::ViBe_impl(size_t N, size_t R, size_t nMin, size_t nSigma) :
    m_N(N),
    m_R(R),
    m_nMin(nMin),
    m_nSigma(nSigma),
	h_background(new uchar[320*240*3*20]),
	d_background(h_background, range<1>(320*240*3*20)),
	d_randoms(h_randoms, range<1>(2)),
    myContext(mySelector, false),
    myQueue(myContext, mySelector) {




}

ViBe_impl::~ViBe_impl() {

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

}


void ViBe_impl::apply(const cv::Mat& oCurrFrame, cv::Mat& oOutputMask) {
    CV_Assert(!oCurrFrame.empty() && oCurrFrame.isContinuous() && oCurrFrame.type()==CV_8UC3);
    oOutputMask.create(oCurrFrame.size(),CV_8UC1); //TODO output is binary, but always stored in a byte (so output values are either '0' or '255')
    size_t frame_size = oCurrFrame.cols * oCurrFrame.rows;
h_randoms[0] = rand()%16;
h_randoms[1] = rand()%20;
{

buffer<uchar, 1> d_image(oCurrFrame.data, range<1>(frame_size * 3));
buffer<uchar, 1> d_output(oOutputMask.data, range<1>(frame_size));
// create a command_group to issue commands to the queue
myQueue.submit([&](handler& cgh) {
    // request access to the buffer
    auto image = d_image.get_access<access::mode::read>(cgh);
    auto output = d_output.get_access<access::mode::write>(cgh);
    auto background = d_background.get_access<access::mode::read_write>(cgh);
	auto randoms = d_randoms.get_access<access::mode::read>(cgh);
    // enqueue a prallel_for task
    cgh.parallel_for<class simple_test>(range<1>(320*240), [=](id<1> idx) {
        //output[idx] = image[idx[0]*3];
        int nb_matchs = 0;
        int dist = 0;

        for(int j = 0; j < 20; ++j) {
            dist = (background[idx[0] * 3 * 20 + j * 3] - image[idx[0] * 3]) * (background[idx[0] * 3 * 20 + j * 3] - image[idx[0] * 3]) +
            (background[idx[0] * 3 * 20 + j * 3 + 1] - image[idx[0] * 3 + 1]) * (background[idx[0] * 3 * 20 + j * 3 + 1] - image[idx[0] * 3 + 1]) +
            (background[idx[0] * 3 * 20 + j * 3 + 2] - image[idx[0] * 3 + 2]) * (background[idx[0] * 3 * 20 + j * 3 + 2] - image[idx[0] * 3 + 2]);
            if(dist <= 20*40)
                nb_matchs++;
        }
        if(nb_matchs >= 1) {
			if(randoms[0] == 0) {
				 background[idx[0] * 3 * 20 + 3 * randoms[1]] = image[idx[0] * 3];
				 background[idx[0] * 3 * 20 + 3 * randoms[1] + 1] = image[idx[0] * 3 + 1];
				 background[idx[0] * 3 * 20 + 3 * randoms[1] + 2] = image[idx[0] * 3 + 2];
			 }
            output[idx[0]] = 0;
        } else {
            output[idx[0]] = 255;
        }

    }); // end of the kernel function
}); // end of our commands for this queue
}

//    filtre median
//    cv::medianBlur(oOutputMask, oOutputMask, 9);
}
