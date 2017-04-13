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

    diff_loss.resize(num);
    bound_loss.resize(num);

    alpha = this->layer_param_.linear_loss_param().alpha();
}

template <typename Dtype>
void LinearLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top) {

    vector<int> shape;
    shape.resize(1);
    shape[0] = 1;

    top[0]->Reshape(shape);
}


template <typename Dtype>
void LinearLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();

  Dtype* loss = top[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;


  for (int i =0 ; i < num; i++) {
    diff_loss[i] = 0;
    bound_loss[i] = 0;
  }

  for (int i = 0; i < num; i++) {
    tgt_dot[i] = caffe_cpu_dot(dim, bottom_data+i*dim, bottom_data+(num-1)*dim);
    l2_squre[i] = caffe_cpu_dot(dim, bottom_data+i*dim, bottom_data+i*dim);
    l2_norm[i] = sqrt(l2_squre[i]);
  }
  for (int i = 0; i < num; i++) {
    tgt_dot[i] = tgt_dot[i]/l2_norm[i]/l2_norm[num-1];
  }

  sum_bound_loss = sum_diff_loss = 0;
  for (int i = 0; i < num; i++) {
    if (tgt_dot[num-1-i] < 1-(1+alpha)*i/Dtype(num) )  {
      bound_loss[num-1-i] = 1-(1+alpha)*i/Dtype(num) - tgt_dot[num-1-i];
    }
    else if (tgt_dot[num-1-i] > 1-(1-alpha)*i/Dtype(num) )  {
      bound_loss[num-1-i] = 1-(1-alpha)*i/Dtype(num) - tgt_dot[num-1-i];
    }
    sum_bound_loss += abs(bound_loss[num-1-i]);
  }
  for (int i = 0; i < num-1; i++) {
    diff_loss[num-2-i] = max(Dtype(0), tgt_dot[num-2-i] - tgt_dot[num-1-i] + Dtype(1-0.5)/Dtype(num));
    sum_diff_loss += diff_loss[num-2-i];
  }

  sum_bound_loss = 0;
  loss[0] = (sum_bound_loss+sum_diff_loss)/num;
/*
  for (int i = 0; i < num; i++) {
    cout << tgt_dot[i] << " " << bound_loss[i] << " " << diff_loss[i] << endl;
  }
  getchar();
  */
/*
  cout << "bound_loss: " << sum_bound_loss/num << endl;
  cout << "diff_loss: " << sum_diff_loss/num << endl;
*/

}

template <typename Dtype>
void LinearLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bound_loss = top[0]->mutable_cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;

    caffe_copy(count, bottom_data, bottom_diff);
    for (int i = 0; i < num; i++) {
      caffe_cpu_axpby(dim, Dtype(1)/l2_norm[i]/l2_norm[num-1], bottom_data+(num-1)*dim,
          -tgt_dot[i]/l2_squre[i], bottom_diff+i*dim);
    }

   for (int i = 0; i < num-1; i++) {
      if (bound_loss[i] > 0)
        sign[i] = Dtype(-1);
      else if (bound_loss[i] < 0) 
        sign[i] = Dtype(1);
      else if (bound_loss[i+1] > 0 && diff_loss[i] > 0)
        sign[i] = Dtype(1);
      else
        sign[i] = Dtype(0);
    }

    for (int i = 0; i < num; i++) {
      caffe_scal(dim, diff_loss[i], bottom_diff+i*dim);
    }

}


INSTANTIATE_CLASS(LinearLossLayer);
REGISTER_LAYER_CLASS(LinearLoss);

}  // namespace caffe
