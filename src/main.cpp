
#include "common.hpp"

#define VIBE_N 20       // @@@@ TUNE IF NEEDED
#define VIBE_R 20       // @@@@ TUNE IF NEEDED
#define VIBE_NMIN 2     // @@@@ TUNE IF NEEDED
#define VIBE_NSIGMA 2  // @@@@ TUNE IF NEEDED

int main(int /*argc*/, char** /*argv*/) {
    srand (time(NULL));
    clock_t begin = clock();

    try {
        std::shared_ptr<ViBe> pAlgo = ViBe::createInstance(VIBE_N,VIBE_R,VIBE_NMIN,VIBE_NSIGMA);

        const std::string sBaseDataPath(DATA_ROOT_PATH "/tp1/");
        const std::vector<std::string> vsSequenceNames = {"highway"};
        const std::vector<size_t> vnSequenceSizes = {1700,1099,1499};
        for(size_t nSeqIdx=0; nSeqIdx<vsSequenceNames.size()-2; ++nSeqIdx) {
            std::cout << "\nProcessing sequence '" << vsSequenceNames[nSeqIdx] << "'..." << std::endl;
            const std::string sInitFramePath = sBaseDataPath+vsSequenceNames[nSeqIdx]+"/input/in000001.jpg";
            const cv::Mat oInitFrame = cv::imread(sInitFramePath);
            CV_Assert(!oInitFrame.empty() && oInitFrame.type()==CV_8UC3);
            pAlgo->initialize(oInitFrame);
            cv::Mat oOutputMask(oInitFrame.size(),CV_8UC1,cv::Scalar_<uchar>(0));
            BinClassif oSeqAccumMetrics;
            for(size_t nFrameIdx=1; nFrameIdx<vnSequenceSizes[nSeqIdx]; ++nFrameIdx) {
                std::cout << "\tProcessing input # " << nFrameIdx+1 << " / " << vnSequenceSizes[nSeqIdx] << "..." << std::endl;
                const std::string sCurrFramePath = putf((sBaseDataPath+vsSequenceNames[nSeqIdx]+"/input/in%06d.jpg").c_str(),(int)(nFrameIdx+1));
                const cv::Mat oCurrFrame = cv::imread(sCurrFramePath);
                CV_Assert(!oCurrFrame.empty() && oInitFrame.size()==oCurrFrame.size() && oCurrFrame.type()==CV_8UC3);
                pAlgo->apply(oCurrFrame,oOutputMask);
                CV_Assert(!oOutputMask.empty() && oOutputMask.size()==oCurrFrame.size() && oOutputMask.type()==CV_8UC1);
                const std::string sCurrGTMaskPath = putf((sBaseDataPath+vsSequenceNames[nSeqIdx]+"/groundtruth/gt%06d.png").c_str(),(int)(nFrameIdx+1));
                const cv::Mat oCurrGTMask = cv::imread(sCurrGTMaskPath,cv::IMREAD_GRAYSCALE);
                CV_Assert(!oCurrGTMask.empty() && oCurrGTMask.size()==oCurrFrame.size() && oCurrGTMask.type()==CV_8UC1);
                oSeqAccumMetrics.accumulate(oOutputMask,oCurrGTMask); // we accumulate TP/TN/FP/FN here
                // for display purposes only
                cv::imshow("input",oCurrFrame);
                cv::imshow("gt",oCurrGTMask);
                cv::imshow("output",oOutputMask);
                cv::waitKey(1);
            }


            // @@@@ TODO : using total TP/TN/FP/FN counts in oSeqAccumMetrics, compute and print overall precision, recall, and f-measure here
            double precision = double(oSeqAccumMetrics.nTP) / (double(oSeqAccumMetrics.nTP) + double(oSeqAccumMetrics.nFP));
            double recall = double(oSeqAccumMetrics.nTP) / (double(oSeqAccumMetrics.nTP) + double(oSeqAccumMetrics.nFN));
            double fmeasure = (2 * precision * recall) / (precision + recall);
            std::cout << "Sequence " << vsSequenceNames[nSeqIdx] << " : "<< std::endl;
            std::cout << "  precision : " << precision << std::endl;
            std::cout << "  recall    : " << recall << std::endl;
            std::cout << "  f-measure : " << fmeasure << std::endl;

        }
        std::cout << "\nAll done." << std::endl;
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Duration : " << elapsed_secs << std::endl;
    }
    catch(const cv::Exception& e) {
        std::cerr << "Caught cv::Exceptions: " << e.what() << std::endl;
    }
    catch(const std::runtime_error& e) {
        std::cerr << "Caught std::runtime_error: " << e.what() << std::endl;
    }
    catch(...) {
        std::cerr << "Caught unhandled exception." << std::endl;
    }
#ifdef _MSC_VER
    system("pause");
#endif //def(_MSC_VER)
    return 0;
}
