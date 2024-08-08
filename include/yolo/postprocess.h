#ifndef _RKNN_YOLOV5_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV5_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>

#include <iomanip>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "common.h"

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.45
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)


int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, BOX_RECT pads, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 DetectResultsGroup *group);

void deinitPostProcess();

int draw_image_detect(cv::Mat &cur_img, std::vector<DetectionBox> &results, int cur_frame_id);

int draw_image_track(cv::Mat &cur_img, std::vector<TrackingBox> &results, int cur_frame_id);

void show_draw_results(cv::Mat &cur_img, std::vector<TrackingBox> &results);


#endif //_RKNN_YOLOV5_DEMO_POSTPROCESS_H_
