#include <algorithm>
#include <vector>

#include "caffe/layers/linear_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>
#include <cstdio>
#include <cstring>
#include <fstream>
using namespace std;

namespace caffe {

template <typename Dtype>
void LinearLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top) {

    LossLayer<Dtype>::LayerSetUp(bottom, top);

    int num = bottom[0]->num();
    tgt_dot.resize(num);
    l2_squre.resize(num);
    l2_norm.resize(num);
    sign.resize(num);

    alpha = this->layer_param_.linear_loss_param().alpha();
}

template <typename Dtype>
void LinearLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top) {

    vector<int> bottom_shape = bottom[0]->shape();
    bottom_shape.resize(1);

    top[0]->Reshape(bottom_shape);
    top[1]->Reshape(bottom_shape);
}


template <typename Dtype>
void LinearLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  Dtype* bound_loss = top[0]->mutable_cpu_data();
  Dtype* diff_loss = top[1]->mutable_cpu_data();

  caffe_set<Dtype>(num, Dtype(0), bound_loss);
  caffe_set<Dtype>(num, Dtype(0), diff_loss);

  for (int i = 0; i < num; i++) {
    tgt_dot[i] = caffe_cpu_dot(dim, bottom_data+i*dim, bottom_data+(num-1)*dim);
    l2_squre[i] = caffe_cpu_dot(dim, bottom_data+i*dim, bottom_data+i*dim);
    l2_norm[i] = sqrt(l2_squre[i]);
  }
  for (int i = 0; i < num; i++) {
    tgt_dot[i] = tgt_dot[i]/l2_norm[i]/l2_norm[num-1];
  }

  sum_bound_loss = 0;
  for (int i = 0; i < num; i++) {
    bound_loss[num-1-i] = max(Dtype(0), (1-(1+alpha)*i/Dtype(num)) - tgt_dot[num-1-i]);
    sum_bound_loss += bound_loss[num-1-i];
  }
  for (int i = 0; i < num-1; i++) {
    diff_loss[num-1-i] = max(Dtype(0), tgt_dot[num-2-i] - tgt_dot[num-1-i] + (1-alpha)/Dtype(num));
  }

}

template <typename Dtype>
void LinearLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bound_loss = top[0]->mutable_cpu_data();
    const Dtype* diff_loss = top[1]->mutable_cpu_data();

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    caffe_copy(count, bottom_data, bottom_diff);
    for (int i = 0; i < num; i++) {
      caffe_cpu_axpby(dim, Dtype(1)/l2_norm[i]/l2_norm[num-1], bottom_data+(num-1)*dim,
          -tgt_dot[i]/l2_squre[i], bottom_diff+i*dim);
    }

    if (sum_bound_loss > 0) {
      for (int i = 0; i < num; i++) {
        sign[i] = (bound_loss[i] > 0) ? Dtype(-1) : Dtype(0);
      }
    }
    else {
      for (int i = 0; i < num; i++) {
        sign[i] = (diff_loss[i] > 0) ? Dtype(1) : Dtype(0);
      }
    }

    for (int i = 0; i < num; i++) {
      caffe_scal(dim, sign[i], bottom_diff+i*dim);
    }

}


INSTANTIATE_CLASS(LinearLossLayer);
REGISTER_LAYER_CLASS(LinearLoss);

}  // namespace caffe
