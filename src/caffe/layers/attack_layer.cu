#include <vector>

#include "caffe/layers/attack_layer.hpp"
//#include "caffe/util/math_functions.hpp"

#include <iostream>
#include <cstdio>
using namespace std;


/*
 * y = 0.5*(tanh(w)+1)
 * |y-x|^2 + c*f(y), f(y) is the attack_loss function
 * the "weights" shuold be initialized by 0, that means search from the original image.
 */

namespace caffe {

template <typename Dtype>
__global__ void myTanHForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = tanh(in[index]);
  }
}

template <typename Dtype>
void AttackLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* L2_loss = top[1]->mutable_cpu_data();
  Dtype* meta = meta_.mutable_gpu_data();
  Dtype* diff = diff_.mutable_gpu_data();
  Dtype* orig = orig_.mutable_gpu_data();
  Dtype* tvalue = tvalue_.mutable_gpu_data();


  if (!has_init_) {
    caffe_gpu_axpby(N_*M_, Dtype(2), bottom_data, Dtype(0), orig);
    caffe_gpu_add_scalar(N_*M_, Dtype(-1), orig);
    caffe_gpu_set(N_*M_, Dtype(0), tvalue);
    has_init_ = true;
  }
  else {
    myTanHForward<Dtype><<<CAFFE_GET_BLOCKS(N_*M_), CAFFE_CUDA_NUM_THREADS>>>(N_*M_, weight, tvalue);
  }

  caffe_gpu_add(N_*M_, orig, tvalue, meta);
  caffe_gpu_mul(N_*M_, orig, tvalue, tvalue);
  caffe_gpu_add_scalar(N_*M_, Dtype(1), tvalue);
  caffe_gpu_div(N_*M_, meta, tvalue, meta);
  caffe_gpu_axpby(N_*M_, Dtype(0.5), meta, Dtype(0), top_data);
  caffe_gpu_add_scalar(N_*M_, Dtype(0.5), top_data);

  caffe_gpu_sub(N_*M_, top_data, bottom_data, diff);
  for (int i = 0; i < N_; i++) {
    caffe_gpu_dot(M_, diff+i*M_, diff+i*M_, L2_loss+i);
    l2loss_[i] = L2_loss[i];
  }
}

template <typename Dtype>
void AttackLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* diff = diff_.mutable_gpu_data();
    Dtype* meta = meta_.mutable_gpu_data();
	Dtype* blob_diff = this->blobs_[0]->mutable_gpu_diff();

    caffe_gpu_mul(N_*M_, meta, meta, meta);
    caffe_gpu_add_scalar(N_*M_, Dtype(-1), meta);
    caffe_gpu_axpy(N_*M_, Dtype(0.5), top_diff, diff);
    caffe_gpu_mul(N_*M_, diff, meta, blob_diff);
    caffe_gpu_scal(N_*M_, Dtype(-1), blob_diff); 

    store_file(top, propagate_down, bottom);

}


INSTANTIATE_LAYER_GPU_FUNCS(AttackLayer);

}  // namespace caffe
