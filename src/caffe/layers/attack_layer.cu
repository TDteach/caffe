#include <vector>

#include "caffe/layers/attack_layer.hpp"

#include <iostream>
#include <cstdio>
using namespace std;

/*
 * x \in [-1,1]
 * y = (tanh(w)+input)/(1-tanh(w)*input) = tanh(w + tanh^{-1}(input) )
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
  Dtype* diff_data = top[1]->mutable_gpu_data();
  Dtype* meta = meta_.mutable_gpu_data();
  Dtype* tvalue = tvalue_.mutable_gpu_data();


  myTanHForward<Dtype><<<CAFFE_GET_BLOCKS(N_*M_), CAFFE_CUDA_NUM_THREADS>>>(N_*M_, weight, tvalue);

  caffe_gpu_add(N_*M_, bottom_data, tvalue, meta);
  caffe_gpu_mul(N_*M_, bottom_data, tvalue, tvalue);
  caffe_gpu_add_scalar(N_*M_, Dtype(1), tvalue);
  caffe_gpu_div(N_*M_, meta, tvalue, top_data);

  caffe_gpu_sub(N_*M_, top_data, bottom_data, diff_data);
}

template <typename Dtype>
void AttackLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* diff_diff = top[1]->gpu_diff();
    Dtype* meta = meta_.mutable_gpu_data();
    Dtype* tvalue = tvalue_.mutable_gpu_data();
	Dtype* blob_diff = this->blobs_[0]->mutable_gpu_diff();

    caffe_gpu_add(N_*M_, top_diff, diff_diff, meta);
    caffe_gpu_mul(N_*M_, top_data, top_data, tvalue);
    caffe_gpu_scal(N_*M_, Dtype(-1), tvalue); 
    caffe_gpu_add_scalar(N_*M_, Dtype(1), tvalue);
    caffe_gpu_mul(N_*M_, meta, tvalue, blob_diff);

//    store_file(top, propagate_down, bottom);

}


INSTANTIATE_LAYER_GPU_FUNCS(AttackLayer);

}  // namespace caffe
