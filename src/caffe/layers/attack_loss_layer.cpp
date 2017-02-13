#include <algorithm>
#include <vector>

#include "caffe/layers/attack_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>
#include <cstdio>
#include <cstring>
#include <fstream>
using namespace std;

namespace caffe {

template <typename Dtype>
void AttackLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top) {

    LossLayer<Dtype>::LayerSetUp(bottom, top);

    vector<int> bottom_shape = bottom[0]->shape();

//	  tgt_ = this->layer_param_.attack_loss_param().tgt_class();
    string filename = this->layer_param_.attack_loss_param().source();
    FILE *fp = fopen(filename.c_str(),"r");
    fscanf(fp, "%d", &tgt_);
    fclose(fp);

    bottom_shape.resize(1);
    C_.Reshape(bottom_shape);
    cc_.resize(bottom_shape[0]);
    for (int i = 0; i < cc_.size(); i++) {
      cc_[i] = 1000;
    }
    caffe_set<Dtype>(bottom_shape[0], Dtype(1000), C_.mutable_cpu_data());
}

template <typename Dtype>
void AttackLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top) {
    vector<int> bottom_shape = bottom[0]->shape();
    bottom_shape.resize(1);

    top[0]->Reshape(bottom_shape);
}


template <typename Dtype>
void AttackLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* L2_loss = bottom[2]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  Dtype* loss = top[0]->mutable_cpu_data();
  Dtype* C = C_.mutable_cpu_data();

  caffe_set<Dtype>(num, Dtype(0), loss);
  for (int i = 0; i < num; ++i) {
    int max_id = tgt_;
    for (int j = 0; j < dim; ++j) {
      if (bottom_data[i*dim+j] > bottom_data[i*dim+max_id]) {
        max_id = j;
      }
    }
    if (max_id != tgt_) {
        loss[i] = (bottom_data[i*dim+max_id]-bottom_data[i*dim+tgt_])*C[i];
    }
    else {
      ++cc_[i];
      if (cc_[i] > 19) {
        if (C[i] > 1e-5) C[i] *= 0.1;
        LOG(INFO) << C[i] ;
        cc_[i] = 0;
      }
    }
    loss[i] += L2_loss[i];

  }

}

template <typename Dtype>
void AttackLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype* l2_diff = bottom[2]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;
    const Dtype* C = C_.cpu_data();

    caffe_set<Dtype>(num, Dtype(1), l2_diff);
	  caffe_set<Dtype>(count, Dtype(0), bottom_diff);
    for (int i = 0; i < num; i++) {
        int max_id = tgt_;
        for (int j = 0; j < dim; j++) {
            if (bottom_data[i*dim+j] > bottom_data[i*dim+max_id]) {
                max_id = j;
            }
        }
        if (max_id != tgt_){
            bottom_diff[i*dim+max_id] = C[i];
            bottom_diff[i*dim+tgt_] = -C[i];
        }
        else {
          l2_diff[i] = 0;
        }
/*
        if (l2_diff[i] < 0.5) {
        for (int j = 0; j < dim; j++) {
          cout << bottom_data[i*dim+j] << " ";
        }
        cout << endl;
        getchar();
        }
        */
	  } 
}


INSTANTIATE_CLASS(AttackLossLayer);
REGISTER_LAYER_CLASS(AttackLoss);

}  // namespace caffe
