
#include "common.hpp"


// local implementation for ViBe segmentation algorithm
struct ViBe_impl : ViBe {
    ViBe_impl(size_t N, size_t R, size_t nMin, size_t nSigma);
    virtual void initialize(const cv::Mat& oInitFrame);
    virtual void apply(const cv::Mat& oCurrFrame, cv::Mat& oOutputMask);
    const size_t m_N; //< internal ViBe parameter; number of samples to keep in each pixel model
    const size_t m_R; //< internal ViBe parameter; maximum color distance allowed between RGB samples for matching
    const size_t m_nMin; //< internal ViBe parameter; required number of matches for background classification
    const size_t m_nSigma; //< internal ViBe parameter; model update rate

    // new methods
    bool checkDescriptor(const cv::Mat &currentArea, int coo);
    bool checkIntensity(const cv::Vec3b& curPix, int coo);
    bool L2distance(const cv::Vec3b& pix, const cv::Vec3b& samples);
    void applyMorpho(cv::Mat& oOutputMask);
    int computeLBP(const cv::Mat& area);
    int distanceLBP(cv::Vec3b pix, cv::Vec3b neighbour);

    // @@@@ ADD ALL REQUIRED DATA MEMBERS FOR BACKGROUND MODEL HERE
    std::vector<std::vector<cv::Vec3b>> intensity; //intensity background model
    std::vector<std::vector<int>> descriptors; //descriptors background model
};

std::shared_ptr<ViBe> ViBe::createInstance(size_t N, size_t R, size_t nMin, size_t nSigma) {
    return std::shared_ptr<ViBe>(new ViBe_impl(N,R,nMin,nSigma));
}

ViBe_impl::ViBe_impl(size_t N, size_t R, size_t nMin, size_t nSigma) :
    m_N(N),
    m_R(R),
    m_nMin(nMin),
    m_nSigma(nSigma) {}


void ViBe_impl::initialize(const cv::Mat& oInitFrame) {
    CV_Assert(!oInitFrame.empty() && oInitFrame.isContinuous() && oInitFrame.type()==CV_8UC3);

    // hint: we work with RGB images, so the type of one pixel is a "cv::Vec3b"! (i.e. three uint8_t's are stored per pixel)
    //loop over the initial frame (except the outer border)
    intensity.clear();
//    descriptors.clear();
    for(int i = 1; i < oInitFrame.rows-1; i++)
    {
        for(int j = 1; j < oInitFrame.cols-1; ++j)
        {
            std::vector<cv::Vec3b> tmp; //contain intensity samples for 1 pixel
            //neighbours pixels
            cv::Vec3b neighbours[] = {oInitFrame.at<cv::Vec3b>(i-1,j-1), oInitFrame.at<cv::Vec3b>(i-1,j), oInitFrame.at<cv::Vec3b>(i-1,j+1), oInitFrame.at<cv::Vec3b>(i,j-1),
                                  oInitFrame.at<cv::Vec3b>(i,j+1), oInitFrame.at<cv::Vec3b>(i+1,j-1), oInitFrame.at<cv::Vec3b>(i+1,j), oInitFrame.at<cv::Vec3b>(i+1,j+1)};
            //loop to put m_N random neighbours in the tmp vector
            for(int k = 0; k < m_N; ++k)
                tmp.push_back(neighbours[rand() % 8]);
            //add the sample vector to the background intensity model
            intensity.push_back(tmp);

            //compute pixel LBP value
//            int LBPvalue= computeLBP(oInitFrame(cv::Rect(j-1, i-1, 3, 3)));
            //initialize LBP samples for the pixel
//            descriptors.push_back(std::vector<int>(m_N, LBPvalue));
        }
    }
}


//apply morphological operation : opening
void ViBe_impl::applyMorpho(cv::Mat& oOutputMask){
    int erosion_size = 2;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                          cv::Size( erosion_size + 1,  erosion_size + 1),
                          cv::Point(erosion_size, erosion_size));

    cv::erode(oOutputMask, oOutputMask, element);
    erosion_size = 4;
    element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                          cv::Size( erosion_size + 1,  erosion_size + 1),
                          cv::Point(erosion_size, erosion_size));
    cv::dilate(oOutputMask, oOutputMask, element);
}

//check if two pixels are quite similar (L2 distance)
bool ViBe_impl::L2distance(const cv::Vec3b& pix, const cv::Vec3b& samples){
    return (sqrt(pow(pix.val[0]-samples.val[0],2) + pow(pix.val[1]-samples.val[1],2) + pow(pix.val[2]-samples.val[2],2)) <= m_R);
}

//get pixel intensity from RGB value
float rgb2gray(cv::Vec3b pix){
    return 0.299 * pix.val[2] * 0.587 * pix.val[1] * 0.114 * pix.val[0];
}

//return 1 if the descriptor values are closed, and 0 otherwise
int ViBe_impl::distanceLBP(cv::Vec3b pix, cv::Vec3b neighbour){
    return(abs(rgb2gray(pix)- rgb2gray(neighbour)) <= 0.365 * rgb2gray(pix));
}

//compute a LBP descriptor for one pixel
int ViBe_impl::computeLBP(const cv::Mat& area){
    CV_Assert(area.cols == 3 && area.rows == 3);
    cv::Vec3b centerPixel = area.at<cv::Vec3b>(1,1);
    int counter = 0, res = 0;
    for(int i = 0; i < area.rows; ++i)
        for(int j = 0; j < area.cols; ++j)
            if(!(i==1&&j==1))
                res += distanceLBP(centerPixel, area.at<cv::Vec3b>(i,j)) * pow(counter++,2);
    return res;
}

//get Hamming distance between two integers
int hammingDist(int a, int b){
    int val = a ^ b;
    int dist = 0;
    while(val)
    {
      ++dist;
      val &= val - 1;
    }
    return dist;
}

//check if a pixel is background with LBP descriptors
bool ViBe_impl::checkDescriptor(const cv::Mat& currentArea, int coo){
    int res = computeLBP(currentArea), counter = 0, k = 0;
    //count how many descriptors are close to the current pixel
    while(counter < m_nMin && k < m_N)
        if(hammingDist(res, descriptors.at(coo).at(k++)) <= 3) //if less than 3 bits are different, it is ok
            counter++;
    //update background descriptor model
    if(!(rand() % m_nSigma))
        descriptors.at(coo).at(rand() % m_N) = res;
    return (counter == m_nMin);
}

//check if a pixel is background with intensity
bool ViBe_impl::checkIntensity(const cv::Vec3b& curPix, int coo){
    int nbOk = 0, counter = 0;
    //count how many samples are close to the current pixel
    while (nbOk < m_nMin && counter < m_N)
        if(L2distance(intensity.at(coo).at(counter++), curPix))
            nbOk++;
    return (nbOk == m_nMin);
}



void ViBe_impl::apply(const cv::Mat& oCurrFrame, cv::Mat& oOutputMask) {
    CV_Assert(!oCurrFrame.empty() && oCurrFrame.isContinuous() && oCurrFrame.type()==CV_8UC3);
    oOutputMask.create(oCurrFrame.size(),CV_8UC1); // output is binary, but always stored in a byte (so output values are either '0' or '255')

    int coo = 0;
    //loop over the current frame
    for(int i = 1; i < oCurrFrame.rows-1; i++)
    {
        for(int j = 1; j < oCurrFrame.cols-1; ++j)
        {
            coo = (i-1)*(oCurrFrame.cols-2)+j-1;    //compute coordinate (i,j) for the vector background model (1 dimension)
            cv::Vec3b curPix = oCurrFrame.at<cv::Vec3b>(i,j); //current pixel value
            cv::Mat roi(oCurrFrame(cv::Rect(j-1, i-1, 3, 3))); //pixel neighbourhood 3x3

            //pixel is background
            if(checkIntensity(curPix, coo))// && checkDescriptor(roi, coo))
            {
                oOutputMask.at<uchar>(i,j) = 0;

                //update background intensity model
                //add the new sample with m_nSigma probability
                if(!(rand() % m_nSigma))
                    intensity.at(coo).at(rand() % m_N) = curPix;

                //update neighbours
                if(i != 1 && i != oCurrFrame.rows -2 && j != 1 && j != oCurrFrame.cols -2)
                {
                    //only with a 1/16 probability
                    if(!(rand()%m_nSigma)){
                        int neighbours = rand() % 8; //get 1 random neighbours to be updated
                        if(neighbours == 0)
                            intensity.at((i-2)*(oCurrFrame.cols-2)+j-2).at(rand()%m_N) = curPix;
                        else if (neighbours == 1)
                            intensity.at((i-2)*(oCurrFrame.cols-2)+j-1).at(rand()%m_N) = curPix;
                        else if (neighbours == 2)
                            intensity.at((i-2)*(oCurrFrame.cols-2)+j).at(rand()%m_N) = curPix;
                        else if (neighbours == 3)
                            intensity.at((i-1)*(oCurrFrame.cols-2)+j).at(rand()%m_N) = curPix;
                        else if (neighbours == 4)
                            intensity.at((i-1)*(oCurrFrame.cols-2)+j-2).at(rand()%m_N) = curPix;
                        else if (neighbours == 5)
                            intensity.at((i)*(oCurrFrame.cols-2)+j-2).at(rand()%m_N) = curPix;
                        else if (neighbours == 6)
                            intensity.at((i)*(oCurrFrame.cols-2)+j-1).at(rand()%m_N) = curPix;
                        else if (neighbours == 7)
                            intensity.at((i)*(oCurrFrame.cols-2)+j).at(rand()%m_N) = curPix;
                    }
                }
            }
            //pixel is foreground
            else {
                oOutputMask.at<uchar>(i,j) = 255;
            }
        }
    }

    //filtre median
    cv::medianBlur(oOutputMask, oOutputMask, 9);

    //morphologic operation
//    applyMorpho(oOutputMask);
}
