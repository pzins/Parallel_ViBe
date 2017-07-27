#include <CL/sycl.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>


int main() {
    using namespace cl::sycl;
    cv::Mat shore_mask = cv::imread("img.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    uchar data[320*240]; // initialize data to be worked on
    // By including all the SYCL work in a {} block, we ensure
    // all SYCL tasks must complete before exiting the block
    {
        // create a queue to enqueue work to
        queue myQueue;
        // wrap our data variable in a buffer
        buffer<uchar, 1> imgBuf((uchar*)shore_mask.data, range<1>(320*240));
        buffer<uchar, 1> resultBuf(data, range<1>(320*240));
        // create a command_group to issue commands to the queue
        myQueue.submit([&](handler& cgh) {
            // request access to the buffer
            auto writeResult = resultBuf.get_access<access::mode::write>(cgh);
	    auto readResult = imgBuf.get_access<access::mode::read>(cgh);
            // enqueue a prallel_for task
            cgh.parallel_for<class simple_test>(range<1>(320*240), [=](id<1> idx) {
                writeResult[idx] = readResult[idx[0]];
            }); // end of the kernel function
        }); // end of our commands for this queue
    } // end of scope, so we wait for the queued work to complete
    // print result
    for (unsigned long int i = 0; i < 320*240; i++)
	std::cout<<"data["<<i<<"] = "<<(int)data[i]<<std::endl;
    return 0;
}
