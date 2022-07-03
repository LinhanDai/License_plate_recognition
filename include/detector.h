//
// Created by linhan on 2021/2/3.
//

#ifndef DETECTOR_H
#define DETECTOR_H

#include <iostream>
#include <chrono>
#include <map>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include "class_detector.h"
#include "class_timer.hpp"
#include "ocrStruct.h"

namespace OcrSystem
{

class YoloDetector
{
public:
    YoloDetector(std::string configPath);
    ~YoloDetector(){}
    void getDetectorResult(std::vector<cv::Mat> &batchImg, std::vector<OcrSystem::BatchOcrResult> &batchOcrRes);

private:
    void paramInit(std::string configFile);
    void yoloInitModel();

private:
    int mNetType;
    float mDetectThresh;
    std::string mFileModelCfg;
    std::string mFileModelWeights;
    int mInferencePrecison;
    int mMaxBatchSize;
    std::shared_ptr<Detector> detector;
};
}
#endif //DETECTOR_H
