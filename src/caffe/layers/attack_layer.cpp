#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/attack_layer.hpp"
#include "caffe/util/math_functions.hpp"



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
void AttackLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  vector<int> bottom_shape = bottom[0]->shape();
  N_ = bottom[0]->num();
  M_ = bottom[0]->count()/N_;


  meta_.ReshapeLike(*bottom[0]);
  tvalue_.ReshapeLike(*bottom[0]);

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    // Initialize the weights
	  this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(bottom_shape));

    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.attack_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void AttackLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  vector<int> top_shape = bottom[0]->shape();
  top[0]->Reshape(top_shape);

  top_shape.resize(1);
  top_shape[0] = bottom[0]->num();
  top[1]->Reshape(top_shape);

}


float m_fabs(float x)
{
  if (x < 0) x = -x;
  return x;
}

template <typename Dtype>
void AttackLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* diff_data = top[1]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  for (int i = 0; i < N_*M_; i++) {
	  Dtype a = bottom_data[i];
	  Dtype b = tanh(weight[i]);

    top_data[i] = (a+b)/(1+a*b);
  }

  caffe_sub(N_*M_, top_data, bottom_data, diff_data);
}

template <typename Dtype>
void AttackLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* diff_diff = top[1]->cpu_diff();
	  Dtype* blob_diff = this->blobs_[0]->mutable_cpu_diff();

    // Gradient with respect to weight
	  for (int i = 0; i < N_*M_; i++) {
		  blob_diff[i] = (top_diff[i]+diff_diff[i])*(1-top_data[i]*top_data[i]);
	  }

//    store_file(top, propagate_down, bottom);
}


template <typename Dtype>
void AttackLayer<Dtype>::store_file(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    bool need = false;
    const Dtype* l2_diff = top[1]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();


    for (int i = 0; i < N_; i++) {
      if (l2_diff[i] < 0.5 && l2loss_[i] < best_[i] ) {
        best_[i] = l2loss_[i];
        need = true;
      }
    }
    
    if (need) {
      FILE *fp = fopen("/home/tdteach/workspace/AT-ResNet-caffe/rst.txt","w");
      for (int i = 0; i < N_; i++) {
        fprintf(fp, "%f\n", best_[i]);
      }
      for (int i = 0; i < M_; i++) {
        fprintf(fp, "%f ", bottom_data[i]);
      }
      fprintf(fp, "\n");
      fclose(fp);
    }
}

#ifdef CPU_ONLY
STUB_GPU(AttackLayer);
#endif

INSTANTIATE_CLASS(AttackLayer);
REGISTER_LAYER_CLASS(Attack);

}  // namespace caffe
