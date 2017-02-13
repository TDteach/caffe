#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/number_series_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
NumberSeriesDataLayer<Dtype>::~NumberSeriesDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void NumberSeriesDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const string& source = this->layer_param_.number_series_data_param().source();
  const int length = this->layer_param_.number_series_data_param().length();
  const int after_k = this->layer_param_.number_series_data_param().after_k();
  const int batch_size = this->layer_param_.number_series_data_param().batch_size();

  int k_times = 1;
  for (int i = 0; i < after_k; i++) k_times *= 10;

  LOG(INFO) << "Opening file " << source;
  origin_.clear();
  FILE *infile = fopen(source.c_str(),"r");
  while (true) {
	  float tmp;
	  if (fscanf(infile,"%f",&tmp) == EOF) break;
      tmp *= k_times;
	  origin_.push_back(int(tmp) % 10);
  }
  fclose(infile);

  CHECK(!origin_.empty()) << "File is empty";

  lines_id_ = 0;
  vector<int> top_shape;
  top_shape.resize(4);
  top_shape[0] = 1;
  top_shape[1] = length;
  top_shape[2] = 1;
  top_shape[3] = 1;

  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;

  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);

  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }
}


// This function is called on prefetch thread
template <typename Dtype>
void NumberSeriesDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int length = this->layer_param_.number_series_data_param().length();
  const int batch_size = this->layer_param_.number_series_data_param().batch_size();

  vector<int> top_shape;
  top_shape.resize(4);
  top_shape[0] = 1;
  top_shape[1] = length;
  top_shape[2] = 1;
  top_shape[3] = 1;
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = origin_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    read_time += timer.MicroSeconds();

    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
	Dtype* ss = this->transformed_data_.mutable_cpu_data();
	for (int i = 0; i < length; i++) {
		ss[i] = origin_[i+lines_id_];
	}
    trans_time += timer.MicroSeconds();

    prefetch_label[item_id] = origin_[lines_id_+length];
    // go to the next iter
    lines_id_++;
    if (lines_id_+length >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(NumberSeriesDataLayer);
REGISTER_LAYER_CLASS(NumberSeriesData);

}  // namespace caffe
#endif  // USE_OPENCV
