#include <CL/sycl.hpp>
#include <iostream>

int main() {
    using namespace cl::sycl;
    int data[1024]; // initialize data to be worked on
    // By including all the SYCL work in a {} block, we ensure
    // all SYCL tasks must complete before exiting the block
    {
        // create a queue to enqueue work to
        queue myQueue;
        // wrap our data variable in a buffer
        buffer<int, 1> resultBuf(data, range<1>(1024));
        // create a command_group to issue commands to the queue
        myQueue.submit([&](handler& cgh) {
            // request access to the buffer
            auto writeResult = resultBuf.get_access<access::mode::write>(cgh);
            // enqueue a prallel_for task
            cgh.parallel_for<class simple_test>(range<1>(1024), [=](id<1> idx) {
                writeResult[idx] = idx[0];
            }); // end of the kernel function
        }); // end of our commands for this queue
    } // end of scope, so we wait for the queued work to complete
    // print result
    for (int i = 0; i < 1024; i++)
    std::cout<<"data["<<i<<"] = "<<data[i]<<std::endl;
    return 0;
}
