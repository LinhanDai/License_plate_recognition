//
// Created by linhan on 2021/2/3.
//

#include "detector.h"


namespace OcrSystem
{
void YoloDetector::paramInit(std::string configPath)
{
    std::string configFile = configPath + "/ocr.yaml";
    cv::FileStorage fs(configFile, cv::FileStorage::READ);
    mNetType = fs["net_type"];
    mDetectThresh = fs["detect_thresh"];
    mFileModelCfg = configPath + "/" + (std::string)fs["file_model_cfg"];
    mFileModelWeights = configPath + "/" + (std::string)fs["file_model_weights"];
    mInferencePrecison = fs["inference_precison"];
    mMaxBatchSize = fs["MAXT_BATCH_SIZE"];
}

void YoloDetector::getDetectorResult(std::vector<cv::Mat> &batchImg, std::vector<OcrSystem::BatchOcrResult> &batchOcrRes)
{
    std::vector<BatchResult> batch_res;
    if(batchImg.size() > mMaxBatchSize)
    {
        std::cerr << "Exceeded the maximum batch size !!!" << std::endl;
        return;
    }
    detector->detect(batchImg, batch_res);
    for (int i=0;i<batchImg.size();++i)
    {
        BatchOcrResult detResult;
        for (const auto &r : batch_res[i])
        {
            OCRBOX box;
            box.top_left = cv::Point(r.rect.x, r.rect.y);
            box.top_right = cv::Point(r.rect.x + r.rect.width, r.rect.y);
            box.bottom_left = cv::Point(r.rect.x, r.rect.y + r.rect.height);
            box.bottom_right = cv::Point(r.rect.x + r.rect.width, r.rect.y + r.rect.height);
            box.score = r.prob;
            detResult.push_back(box);
        }
        batchOcrRes.push_back(detResult);
    }
}

void YoloDetector::yoloInitModel()
{
    Config config_v4_tiny;
    config_v4_tiny.net_type = static_cast<ModelType>(mNetType);
    config_v4_tiny.detect_thresh = mDetectThresh;
    config_v4_tiny.file_model_cfg = mFileModelCfg;
    config_v4_tiny.file_model_weights = mFileModelWeights;
    config_v4_tiny.inference_precison = static_cast<Precision>(mInferencePrecison);
    detector = std::make_shared<Detector>();
	detector->init(config_v4_tiny);
}

YoloDetector::YoloDetector(std::string configFile)
{
    paramInit(configFile);
    yoloInitModel();
}
}

