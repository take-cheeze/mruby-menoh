#include <mruby.h>

#include <opencv2/opencv.hpp>

static mrb_value
load_image_data(mrb_state *mrb, mrb_value self)
{
  char const *in_fn;
  mrb_int width, height;
  mrb_get_args(mrb, "zii", &in_fn, &width, &height);

  cv::Mat mat = cv::imread(in_fn, cv::IMREAD_COLOR);
  if (!mat.data) {
    mrb_raisef(mrb, E_RUNTIME_ERROR, "invalid input image path: %S", mrb_str_new_cstr(mrb, in_fn));
  }

  cv::resize(mat, mat, cv::Size(width, height));
  mat.convertTo(mat, CV_32FC3);
  mat -= cv::Scalar(103.939, 116.779, 123.68);

  assert(mat.channels() == 3);
  std::vector<float> data(mat.channels() * mat.rows * mat.cols);
  for(int y = 0; y < mat.rows; ++y) {
    for(int x = 0; x < mat.cols; ++x) {
      for(int c = 0; c < mat.channels(); ++c) {
        data[c * (mat.rows * mat.cols) + y * mat.cols + x] =
            mat.at<cv::Vec3f>(y, x)[2 - c];
      }
    }
  }
  return mrb_str_new(mrb, (char const*)data.data(), sizeof(float) * data.size());
}

extern "C" void
mrb_mruby_menoh_gem_test(mrb_state *mrb)
{
  mrb_define_method(mrb, mrb->object_class, "load_image_data", load_image_data, MRB_ARGS_REQ(3));
}
