#ifndef CAFFE_ATTACK_LAYER_HPP_
#define CAFFE_ATTACK_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Along with the attack_loss_layer. This layer replace the input image
 * with 0.5(tanh(w)+1), and output the ||0.5(tanh(w)+1)-x||^2 loss. 
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class AttackLayer : public Layer<Dtype> {
 public:
  explicit AttackLayer(const LayerParameter& param)
      : Layer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Attack"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void store_file(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int N_;
  int M_;
  
  Blob<Dtype> meta_;
  Blob<Dtype> tvalue_;
  bool has_init_;
  vector<Dtype> best_;
  vector<Dtype> l2loss_;

};

}  // namespace caffe

#endif  // CAFFE_ATTACK_LAYER_HPP_
