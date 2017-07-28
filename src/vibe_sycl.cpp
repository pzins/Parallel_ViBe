
#include "tp1/common.hpp"
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

    size_t image_size;
    uchar* h_background;
	int h_randoms[2];

    gpu_selector mySelector;
    context myContext;
	queue myQueue;

    buffer<uchar, 1> background_buffer;
	buffer<int, 1> randoms_buffer;
};

std::shared_ptr<ViBe> ViBe::createInstance(size_t N, size_t R, size_t nMin, size_t nSigma) {
    return std::shared_ptr<ViBe>(new ViBe_impl(N,R,nMin,nSigma));
}

ViBe_impl::ViBe_impl(size_t N, size_t R, size_t nMin, size_t nSigma) :
    m_N(N),
    m_R(R),
    m_nMin(nMin),
    m_nSigma(nSigma),
    image_size(320*240),
	h_background(new uchar[image_size*3*20]),
	background_buffer(h_background, range<1>(image_size*3*20)),
	randoms_buffer(h_randoms, range<1>(2)),
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
                h_background[i * 3 * 20 * oInitFrame.cols + (j * 3 * 20 )+ (k * 3)] = neighbours[rand() % 8][0];
                h_background[i * 3 * 20 * oInitFrame.cols + (j * 3 * 20 )+ (k * 3) + 1] = neighbours[rand() % 8][1];
                h_background[i * 3 * 20 * oInitFrame.cols + (j * 3 * 20 )+ (k * 3) + 2] = neighbours[rand() % 8][2];
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
        buffer<uchar, 1> image_buffer(oCurrFrame.data, range<1>(frame_size * 3));
        buffer<uchar, 1> output_buffer(oOutputMask.data, range<1>(frame_size));
        // create a command_group to issue commands to the queue
        myQueue.submit([&](handler& cgh) {

            // request access to the buffer
            auto d_image = image_buffer.get_access<access::mode::read>(cgh);
            auto d_output = output_buffer.get_access<access::mode::write>(cgh);
            auto d_background = background_buffer.get_access<access::mode::read_write>(cgh);
        	auto d_randoms = randoms_buffer.get_access<access::mode::read>(cgh);
            // enqueue a prallel_for task

            cgh.parallel_for<class simple_test>(range<1>(image_size), [=](id<1> idx) {
                int nb_matchs = 0;
                int dist = 0;

                for(int j = 0; j < 20; ++j) {
                    dist = (d_background[idx[0] * 3 * 20 + j * 3] - d_image[idx[0] * 3]) * (d_background[idx[0] * 3 * 20 + j * 3] - d_image[idx[0] * 3]) +
                    (d_background[idx[0] * 3 * 20 + j * 3 + 1] - d_image[idx[0] * 3 + 1]) * (d_background[idx[0] * 3 * 20 + j * 3 + 1] - d_image[idx[0] * 3 + 1]) +
                    (d_background[idx[0] * 3 * 20 + j * 3 + 2] - d_image[idx[0] * 3 + 2]) * (d_background[idx[0] * 3 * 20 + j * 3 + 2] - d_image[idx[0] * 3 + 2]);
                    if(dist <= 20*40)
                        nb_matchs++;
                }
                if(nb_matchs >= 1) {
        			if(d_randoms[0] == 0) {
        				 d_background[idx[0] * 3 * 20 + 3 * d_randoms[1]] = d_image[idx[0] * 3];
        				 d_background[idx[0] * 3 * 20 + 3 * d_randoms[1] + 1] = d_image[idx[0] * 3 + 1];
        				 d_background[idx[0] * 3 * 20 + 3 * d_randoms[1] + 2] = d_image[idx[0] * 3 + 2];
        			 }
                    d_output[idx[0]] = 0;
                } else {
                    d_output[idx[0]] = 255;
                }

            }); // end of the kernel function
        }); // end of our commands for this queue
    } // end of scope, so we wait for the queued work to complete

    //  filtre median
    cv::medianBlur(oOutputMask, oOutputMask, 9);
}
