//
// Created by linhan on 2021/1/20.
//

#ifndef OCRSTRUCT_H
#define OCRSTRUCT_H

#include <iostream>


namespace OcrSystem
{
struct OCRBOX
{
    float score;
    std::string label;
    cv::Point top_left;
    cv::Point top_right;
    cv::Point bottom_right;
    cv::Point bottom_left;
};
typedef std::vector<OCRBOX> BatchOcrResult;
}


#endif //OCR_STRUCT_H
