#include "modelzoo/yolo11n_obb/yolo11n_obb.h"
#include "inference/onnxruntime/onnxruntime.h"
#include <cpptoolkit/log/log.h>
#include "modelzoo/common/img_common.hpp"

#define M_PI 3.14159265358979323846

namespace modelzoo {

namespace {

void covariance_matrix(float w, float h, float r, float &a_val, float &b_val,
                       float &c_val) {
  // 检查输入参数的有效性
  if (w < 0 || h < 0) {
    a_val = b_val = c_val = 0;
    return;
  }
  float a = (w * w) / 12;
  float b = (h * h) / 12;
  float cos_r = cosf(r);
  float sin_r = sinf(r);

  a_val = a * cos_r * cos_r + b * sin_r * sin_r;
  b_val = a * sin_r * sin_r + b * cos_r * cos_r;
  c_val = (a - b) * sin_r * cos_r;
}

float square(float x) {
  return x * x; // 返回输入数的平方
}

cv::Point rotate_point(float x, float y, float theta) {
  float x_new = x * cos(theta) - y * sin(theta);
  float y_new = x * sin(theta) + y * cos(theta);
  return cv::Point2f(x_new, y_new);
}

using YoloTempBox = Yolo11NObb::Box;

float probiou2(const YoloTempBox &obb1, const YoloTempBox &obb2, float eps) {
  float a1, b1, c1;
  float a2, b2, c2;

  covariance_matrix(obb1.box[2], obb1.box[3], obb1.angle, a1, b1, c1);
  covariance_matrix(obb2.box[2], obb2.box[3], obb2.angle, a2, b2, c2);

  float x1 = obb1.box[0] + obb1.box[2] / 2;
  float y1 = obb1.box[1] + obb1.box[3] / 2;
  float x2 = obb2.box[0] + obb2.box[2] / 2;
  float y2 = obb2.box[1] + obb2.box[3] / 2;

  float t1 = ((a1 + a2) * square(y1 - y2) + (b1 + b2) * square(x1 - x2)) /
             ((a1 + a2) * (b1 + b2) - square(c1 + c2) + eps);
  float t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) /
             ((a1 + a2) * (b1 + b2) - square(c1 + c2) + eps);
  float t3 = log(
      ((a1 + a2) * (b1 + b2) - square(c1 + c2)) /
          (4 * sqrt(a1 * b1 - square(c1)) * sqrt(a2 * b2 - square(c2)) + eps) +
      eps);

  float bd = 0.25f * t1 + 0.5f * t2 + 0.5f * t3;
  float hd = sqrtf(1.0f - expf(-fminf(fmaxf(bd, eps), 100.0f)) + eps);

  return 1.0f - hd;
}

void ProbiouNMS(std::vector<YoloTempBox> &yolo_temp_boxes, float nmsThresh) {
  std::sort(yolo_temp_boxes.begin(), yolo_temp_boxes.end(),
            [](const YoloTempBox &a, const YoloTempBox &b) {
              return a.score > b.score;
            });
  std::vector<bool> remove_flags(yolo_temp_boxes.size(), false);

  for (size_t i = 0; i < yolo_temp_boxes.size(); ++i) {
    if (remove_flags[i] || yolo_temp_boxes[i].score == 0)
      continue;

    for (size_t j = i + 1; j < yolo_temp_boxes.size(); ++j) {
      if (remove_flags[j] || yolo_temp_boxes[j].score == 0 ||
          yolo_temp_boxes[i].class_id != yolo_temp_boxes[j].class_id)
        continue;

      if (probiou2(yolo_temp_boxes[i], yolo_temp_boxes[j], 1e-7) > nmsThresh) {
        remove_flags[j] = true;
      }
    }
  }
  std::vector<YoloTempBox> newBoxes;
  for (size_t i = 0; i < yolo_temp_boxes.size(); ++i) {
    if (!remove_flags[i]) {
      newBoxes.push_back(yolo_temp_boxes[i]);
    }
  }
  yolo_temp_boxes = std::move(newBoxes);
}

} // namespace

std::array<cv::Point, 4> Yolo11NObb::Box::ToXYXY() const {
  float w = box[2];
  float h = box[3];
  float x_c = box[0];
  float y_c = box[1];
  float theta = angle;

  std::array<cv::Point2f, 4> corners = {
      cv::Point2f(-w / 2, -h / 2), cv::Point2f(w / 2, -h / 2),
      cv::Point2f(w / 2, h / 2), cv::Point2f(-w / 2, h / 2)};
  std::array<cv::Point, 4> result;
  for (int i = 0; i < 4; i++) {
    cv::Point2f rotated = rotate_point(corners[i].x, corners[i].y, theta);
    rotated.x = x_c + rotated.x;
    rotated.y = y_c + rotated.y;
    result[i] = cv::Point(rotated.x, rotated.y);
  }

  return result;
}

Yolo11NObb::Yolo11NObb() {
  engine_ = std::make_unique<inference::OnnxRuntimeEngine>();
}

Yolo11NObb::~Yolo11NObb() {}

void Yolo11NObb::SetClassNum(int class_num) { class_num_ = class_num; }

int Yolo11NObb::Init(const inference::InferenceParams &params) {
  int ret = engine_->Init(params);
  if (engine_->IsDynamicModel()) {
    LOG_ERROR("must be static model");
    Deinit();
    return -1;
  }

  return ret;
}

void Yolo11NObb::Deinit() { engine_->Deinit(); }

std::string Yolo11NObb::DumpModel() { return engine_->DumpModelInfo(); }

bool Yolo11NObb::IsReady() { return engine_->IsReady(); }

int Yolo11NObb::Warmup() { return engine_->Warmup(); }

int Yolo11NObb::DetectObb(const cv::Mat &img, Result &result) {
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

int Yolo11NObb::Preprocess(const cv::Mat &img) {
  auto i_tensor = engine_->GetInputTensors().at("images");
  auto [dst_img, img_scale] =
      imgutils::LetterBoxPadImage(img, cv::Size(1024, 1024));

  image_info_.raw_size.width = img.cols;
  image_info_.raw_size.height = img.rows;
  image_info_.trans = {1.0f / img_scale, 1.0f / img_scale, 0, 0};

  imgutils::BlobNormalizeFromImage(dst_img, i_tensor.p, i_tensor.data_type);
  return 0;
}

int Yolo11NObb::Postprocess(Result &result) {
  auto o_tensor = engine_->GetOutputTensors().at("output0");
  const auto &o_shape = o_tensor.shape;
  const auto &o_data = o_tensor.p;
  const auto &o_data_type = o_tensor.data_type;

  cv::Mat o_tensor_data = cv::Mat(o_shape[1], o_shape[2], CV_32F, o_data);
  o_tensor_data = o_tensor_data.t();
  auto tensor_p = (float *)o_tensor_data.data;

  int stride_len = o_shape[1];
  int obj_cnt = o_shape[2];

  std::vector<YoloTempBox> yolo_temp_boxes;

  for (int i = 0; i < obj_cnt; i++) {
    float *start_data = tensor_p + i * stride_len;
    cv::Vec4f box = {start_data[0], start_data[1], start_data[2],
                     start_data[3]};

    cv::Mat conf = cv::Mat(1, class_num_, CV_32F, start_data + 4);
    cv::Point class_id;
    double maxClassScore;
    cv::minMaxLoc(conf, nullptr, &maxClassScore, nullptr, &class_id);
    auto *angle = start_data + 4 + class_num_;

    if (maxClassScore > threshold_.det_threshold) {
      yolo_temp_boxes.push_back(
          {box, *angle, (float)maxClassScore, class_id.x});
    }
  }

  ProbiouNMS(yolo_temp_boxes, threshold_.nms_threshold);

  LOG_DEBUG("yolo_temp_boxes size: {}", yolo_temp_boxes.size());
  result = std::move(yolo_temp_boxes);

  for (int i = 0; i < result.size(); i++) {
    auto &res = result[i];
    if (fmod(res.angle, ((float)M_PI)) >= (float)M_PI / 2) {
      std::swap(res.box[2], res.box[3]);
    }
    res.angle = fmod(res.angle, (float)M_PI);
  }

  for (int i = 0; i < result.size(); i++) {
    result[i].box[0] /= image_info_.trans[0];
    result[i].box[1] /= image_info_.trans[1];
    result[i].box[2] /= image_info_.trans[0];
    result[i].box[3] /= image_info_.trans[1];
  }

  return 0;
}

void Yolo11NObb::DrawObb(cv::Mat &g_letter_img, const Result &result) {
  for (int i = 0; i < result.size(); i++) {
    cv::circle(g_letter_img, cv::Point(result[i].box[0], result[i].box[1]), 2,
               cv::Scalar(0, 0, 255), 2);
  }
  for (int i = 0; i < result.size(); i++) {
    cv::polylines(g_letter_img, result[i].ToXYXY(), true, cv::Scalar(0, 255, 0),
                  1);
  }

  for (int i = 0; i < result.size(); i++) {
    cv::putText(g_letter_img,
                std::to_string(result[i].class_id) + " " +
                    std::to_string(result[i].score),
                cv::Point(result[i].box[0], result[i].box[1]),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
  }
}

} // namespace modelzoo