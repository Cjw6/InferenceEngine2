// clang-format off
/*
init yolov11n_pose success, dump:model info:
dynamic model: false
input nums: 1
input: images
TensorDesc(data_type:TensorDataType::FP32, shape:[1, 3, 640, 640], element_size:1228800)
output nums: 2
output: output0
TensorDesc(data_type:TensorDataType::FP32, shape:[1, 116, 8400], element_size:974400)
output: output1
TensorDesc(data_type:TensorDataType::FP32, shape:[1, 32, 160, 160], element_size:819200)
*/
// clang-format on

#include "modelzoo/yolo11n_seg/yolo11n_seg.h"
#include "inference/onnxruntime/onnxruntime.h"
#include "inference/tensor/tensor_helper.h"
#include "inference/utils/assert.h"
#include "inference/utils/exception.h"
#include "inference/utils/log.h"
#include "inference/utils/map.h"
#include "inference/utils/to_string.h"
#include "modelzoo/common/img_common.hpp"

namespace {

int seg_ch = 32;
int seg_w = 160, seg_h = 160;
int net_w = 640, net_h = 640;
float accu_thresh = 0.25, mask_thresh = 0.5;

// struct ImageInfo {
//   cv::Size raw_size;
//   cv::Vec4d trans;
// };

void GetMask(const cv::Mat &mask_info, const cv::Mat &mask_data,
             const modelzoo::Yolo11NSeg::ImageInfo &para, cv::Rect bound,
             cv::Mat &mast_out, std::vector<cv::Point> &mask_countours) {
  cv::Vec4f trans = para.trans;
  int r_x = floor((bound.x * trans[0] + trans[2]) / net_w * seg_w);
  int r_y = floor((bound.y * trans[1] + trans[3]) / net_h * seg_h);
  int r_w =
      ceil(((bound.x + bound.width) * trans[0] + trans[2]) / net_w * seg_w) -
      r_x;
  int r_h =
      ceil(((bound.y + bound.height) * trans[1] + trans[3]) / net_h * seg_h) -
      r_y;
  r_w = MAX(r_w, 1);
  r_h = MAX(r_h, 1);

  LOG_DEBUG("mask bound:{}, {}, {}, {}, {}, {}", r_x, r_y, r_w, r_h, seg_w,
            seg_h);

  if (r_x + r_w > seg_w) // crop
  {
    seg_w - r_x > 0 ? r_w = seg_w - r_x : r_x -= 1;
  }
  if (r_y + r_h > seg_h) {
    seg_h - r_y > 0 ? r_h = seg_h - r_y : r_y -= 1;
  }
  std::vector<cv::Range> roi_rangs = {cv::Range(0, 1), cv::Range::all(),
                                      cv::Range(r_y, r_h + r_y),
                                      cv::Range(r_x, r_w + r_x)};
  cv::Mat temp_mask = mask_data(roi_rangs).clone();
  cv::Mat protos = temp_mask.reshape(0, {seg_ch, r_w * r_h});
  cv::Mat matmul_res = (mask_info * protos).t();
  cv::Mat masks_feature = matmul_res.reshape(1, {r_h, r_w});
  cv::Mat dest;
  cv::exp(-masks_feature, dest); // sigmoid
  dest = 1.0 / (1.0 + dest);
  int left = floor((net_w / seg_w * r_x - trans[2]) / trans[0]);
  int top = floor((net_h / seg_h * r_y - trans[3]) / trans[1]);
  int width = ceil(net_w / seg_w * r_w / trans[0]);
  int height = ceil(net_h / seg_h * r_h / trans[1]);
  cv::Mat mask;
  cv::resize(dest, mask, cv::Size(width, height));
  mast_out = mask(bound - cv::Point(left, top)) > mask_thresh;

  LOG_DEBUG("l:{} t:{} w:{} h:{}", left, top, width, height);
  std::cout  << "mask_out.type()"<< mast_out.type() << std::endl;
  LOG_DEBUG("mask_out.size(): {}", cpputils::ToString(mast_out.size()));
 
  // 获得边界框
  cv::Mat real_img =
      cv::Mat::zeros(para.raw_size.height, para.raw_size.width, CV_8UC1);

  auto roi = real_img(bound);
  mast_out.copyTo(roi);
  
  // real_img(cv::Rect(left, top, width, height)).setTo(cv::Scalar(255), mast_out);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(real_img, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  mask_countours.clear();
  if (contours.size() > 0) {
    int idx = 0;
    for (int i = 0; i < contours.size(); i++) {
      if (contours[i].size() > contours[idx].size()) {
        idx = i;
      }
    }
    mask_countours = contours[idx];
  }
}

void DecodeOutput(cv::Mat &output0, cv::Mat &output1,
                  modelzoo::Yolo11NSeg::ImageInfo para,
                  std::vector<modelzoo::Yolo11NSeg::ResultObj> &output,
                  int class_cnt) {
  auto trans = para.trans;
  LOG_DEBUG("start decode output, trans:{}, {}, {}, {}", trans[0], trans[1],
            trans[2], trans[3]);
  output.clear();
  std::vector<int> class_ids;
  std::vector<float> accus;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<float>> masks;
  int data_width = class_cnt + 4 + 32;
  int rows = output0.rows;
  float *pdata = (float *)output0.data;
  for (int r = 0; r < rows; ++r) {
    cv::Mat scores(1, class_cnt, CV_32FC1, pdata + 4);
    cv::Point class_id;
    double max_socre;
    minMaxLoc(scores, 0, &max_socre, 0, &class_id);
    if (max_socre >= accu_thresh) {
      masks.push_back(
          std::vector<float>(pdata + 4 + class_cnt, pdata + data_width));

      float w = pdata[2] / para.trans[0];
      float h = pdata[3] / para.trans[1];
      int left = MAX(
          int((pdata[0] - para.trans[2]) / para.trans[0] - 0.5 * w + 0.5), 0);
      int top = MAX(
          int((pdata[1] - para.trans[3]) / para.trans[1] - 0.5 * h + 0.5), 0);
      class_ids.push_back(class_id.x);
      accus.push_back(max_socre);
      boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
    }
    pdata += data_width; // next line
  }
  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, accus, accu_thresh, mask_thresh, nms_result);
  for (int i = 0; i < nms_result.size(); ++i) {
    int idx = nms_result[i];
    // boxes[idx] =
    // boxes[idx] & cv::Rect(0, 0, para.raw_size.width, para.raw_size.height);
    modelzoo::Yolo11NSeg::ResultObj result = {class_ids[idx], accus[idx],
                                              boxes[idx]};
    GetMask(cv::Mat(masks[idx]).t(), output1, para, boxes[idx], result.mask,
            result.mask_countours);
    output.push_back(result);
  }
}

} // namespace

namespace modelzoo {

Yolo11NSeg::Yolo11NSeg() {
  engine_ = std::make_unique<inference::OnnxRuntimeEngine>();
}

Yolo11NSeg::~Yolo11NSeg() {}

int Yolo11NSeg::Init(const inference::InferenceParams &params) {
  int ret = engine_->Init(params);
  if (engine_->IsDynamicModel()) {
    LOG_ERROR("must be static model");
    Deinit();
    return -1;
  }

  return ret;
}

void Yolo11NSeg::Deinit() { engine_->Deinit(); }

std::string Yolo11NSeg::DumpModel() { return engine_->DumpModelInfo(); }

bool Yolo11NSeg::IsReady() { return engine_->IsReady(); }

int Yolo11NSeg::Warmup() { return engine_->Warmup(); }

int Yolo11NSeg::Segment(const cv::Mat &img, Result &result) {
  result.clear();
  if (img.empty() || img.type() != CV_8UC3) {
    LOG_ERROR("invalid image");
    return -1;
  }

  int ret = Preprocess(img);
  if (ret != 0) {
    LOG_ERROR("preprocess failed: {}", ret);
    return -2;
  }

  ret = engine_->Run();
  if (ret != 0) {
    LOG_ERROR("run model failed: {}", ret);
    return -3;
  }

  ret = Postprocess(result);
  if (ret != 0) {
    LOG_ERROR("postprocess failed: {}", ret);
    return -4;
  }

  return 0;
}

int Yolo11NSeg::Preprocess(const cv::Mat &img) {
  auto i_tensor = engine_->GetInputTensors().at("images");
  auto [dst_img, img_scale] =
      imgutils::LetterBoxPadImage(img, cv::Size(640, 640));
  // img_scales_ = img_scale;
  img_info_.raw_size.width = img.cols;
  img_info_.raw_size.height = img.rows;
  img_info_.trans = {1.0f / img_scale, 1.0f / img_scale, 0, 0};

  imgutils::BlobNormalizeFromImage(dst_img, i_tensor.p, i_tensor.data_type);
  return 0;
}

int Yolo11NSeg::Postprocess(Result &result) {
  auto output_0 = engine_->GetOutputTensors().at("output0");
  auto output_1 = engine_->GetOutputTensors().at("output1");

  auto data_shape = output_0.shape;
  cv::Mat output0 = cv::Mat(cv::Size((int)data_shape[2], (int)data_shape[1]),
                            CV_32F, output_0.p)
                        .t();
  auto mask_shape = output_1.shape;
  std::vector<int> mask_sz = {1, (int)mask_shape[1], (int)mask_shape[2],
                              (int)mask_shape[3]};
  cv::Mat output1 = cv::Mat(mask_sz, CV_32F, output_1.p);
  // ImageInfo img_info = {cv::Size(640, 640),
  // {1.0f / img_scales_ , 1.0f / img_scales_, 0, 0}};

  DecodeOutput(output0, output1, img_info_, result, 80);
  return 0;
}

void Yolo11NSeg::DrawResult(cv::Mat &img, Result &result,
                            std::vector<cv::Scalar> color) {
  cv::Mat mask = img.clone();
  for (int i = 0; i < result.size(); i++) {
    int left, top;
    left = result[i].bound.x;
    top = result[i].bound.y;
    int color_num = i;
    rectangle(img, result[i].bound, color[result[i].id], 8);
    if (result[i].mask.rows && result[i].mask.cols > 0) {
      mask(result[i].bound).setTo(color[result[i].id], result[i].mask);
    }
    std::string label = std::format("{}:{:.2f}", result[i].id, result[i].accu);
    putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 2,
            color[result[i].id], 4);
    auto &mask_countours = result[i].mask_countours;
    for (auto &point : mask_countours) {
      cv::circle(mask, point, 2, cv::Scalar(0, 0, 255), -1);
    }
  }
  addWeighted(img, 0.6, mask, 0.4, 0, img); // add mask to src
}

} // namespace modelzoo