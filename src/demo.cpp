#include "crnn.h"
#include "detector.h"
#include "ocrStruct.h"
#include "CvxText.h"


int main()
{
    std::vector<cv::String> images;
    std::string configFile = "../configs/";
    OcrSystem::YoloDetector det = OcrSystem::YoloDetector(configFile);
    OcrSystem::CrnnRec rec = OcrSystem::CrnnRec(configFile);
    cv::String path("../data/*.jpg");
    cv::glob(path, images);
    std::vector<cv::Mat> batchImg;
    bool testSpeedFlag = false;      //if you want to test end-to-end batch = 1 speed, you can set the testSpeedFlag = true
    bool bacthInferenceFlag = false;    //if you want to Multi-batch inference, the flag bit is 1
    if (testSpeedFlag)
    {
        for (auto image: images)
        {
            std::vector<cv::Mat> batchTest;
            std::vector<OcrSystem::BatchOcrResult> batchOcrRes;;
            cv::Mat img = cv::imread(image, cv::IMREAD_COLOR);
            double start = (double) cv::getTickCount();
            batchTest.push_back(img);
            det.getDetectorResult(batchTest, batchOcrRes);
            rec.getRecResult(batchTest, batchOcrRes);
            double end = (double) cv::getTickCount();
            std::cout << "CT:" << (end - start) * 1000 / cv::getTickFrequency() << std::endl;
        }
        rec.releaseResource();
        exit(0);
    }

    if (bacthInferenceFlag)
    {
        //maximum 20 batch inputs are supported. If you want to adjust, you can set "batch" parameter in Yolo's configuration file
        for (auto image: images)
        {
            cv::Mat img = cv::imread(image, cv::IMREAD_COLOR);
            batchImg.push_back(img);
        }
        std::vector<OcrSystem::BatchOcrResult> batchOcrRes;
        det.getDetectorResult(batchImg, batchOcrRes);
        rec.getRecResult(batchImg, batchOcrRes);
        //need free memory to prevent memory leakage
        rec.releaseResource();

        //set font and show chinese in image
        CvxText text("../configs/simhei.ttf");
        cv::Scalar fontSize{ 25, 0.5, 0.1, 0 };         // (font size, invalid, character spacing, invalid}
        text.setFont(nullptr, &fontSize, nullptr, 0);
        //visualization result in img
        for (int i = 0; i < batchImg.size(); i++)
        {
            cv::Mat srcImg;
            batchImg[i].copyTo(srcImg);
            //It is possible to detect multiple text boxes
            for(int j = 0; j < batchOcrRes[i].size(); j++)
            {
                std::stringstream stream;
                stream << std::fixed << std::setprecision(2) << batchOcrRes[i][j].label << " score:" << batchOcrRes[i][j].score;
                std::string streamstr = stream.str();
                const char *data = streamstr.c_str();
                cv::rectangle(srcImg, batchOcrRes[i][j].top_left, batchOcrRes[i][j].bottom_right, cv::Scalar(255, 0, 0), 2);
                wchar_t *w_str;
                text.toWchar(const_cast<char *&>(data), w_str);
                text.putText(srcImg, w_str, cv::Point(batchOcrRes[i][j].top_left.x,batchOcrRes[i][j].top_left.y - 5), cv::Scalar(0, 0, 255));
            }
            cv::imshow("img", srcImg);
            cv::waitKey(0);
        }
    }
    else
    {
        for (auto image: images)
        {
            cv::Mat img = cv::imread(image, cv::IMREAD_COLOR);
            batchImg.clear();
            batchImg.push_back(img);
            std::vector<OcrSystem::BatchOcrResult> batchOcrRes;
            det.getDetectorResult(batchImg, batchOcrRes);
            rec.getRecResult(batchImg, batchOcrRes);
            //set font and show chinese in image
            CvxText text("../configs/simhei.ttf");
            cv::Scalar fontSize{ 25, 0.5, 0.1, 0 };         // (font size, invalid, character spacing, invalid}
            text.setFont(nullptr, &fontSize, nullptr, 0);
            //visualization result in img
            for (int i = 0; i < batchImg.size(); i++)
            {
                cv::Mat srcImg;
                batchImg[i].copyTo(srcImg);
                //It is possible to detect multiple text boxes
                for(int j = 0; j < batchOcrRes[i].size(); j++)
                {
                    std::stringstream stream;
                    stream << std::fixed << std::setprecision(2) << batchOcrRes[i][j].label << " score:" << batchOcrRes[i][j].score;
                    std::string streamstr = stream.str();
                    const char *data = streamstr.c_str();
                    cv::rectangle(srcImg, batchOcrRes[i][j].top_left, batchOcrRes[i][j].bottom_right, cv::Scalar(255, 0, 0), 2);
                    wchar_t *w_str;
                    text.toWchar(const_cast<char *&>(data), w_str);
                    text.putText(srcImg, w_str, cv::Point(batchOcrRes[i][j].top_left.x,batchOcrRes[i][j].top_left.y - 5), cv::Scalar(0, 0, 255));
                }
                cv::imshow("img", srcImg);
                cv::waitKey(0);
            }
        }
        rec.releaseResource();
    }
}
