#ifndef CAFFE_LINEAR_LOSS_LAYER_HPP_
#define CAFFE_LINEAR_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class LinearLossLayer : public LossLayer<Dtype> {
 public:
  explicit LinearLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LinearLoss"; }
  virtual inline int ExactNumBottomBlobs() const {return 2; }

 protected:
  /// @copydoc EuclideanLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype alpha;
  Dtype sum_bound_loss;
  Dtype sum_diff_loss;

  vector<Dtype> tgt_dot;
  vector<Dtype> l2_squre;
  vector<Dtype> l2_norm;
  vector<Dtype> sign;

  vector<Dtype> diff_loss;
  vector<Dtype> bound_loss;

};

}  // namespace caffe

#endif  // CAFFE_ATTACK_LOSS_LAYER_HPP_
