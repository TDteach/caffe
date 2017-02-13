#ifndef CAFFE_NUMBER_SERIES_DATA_LAYER_HPP_
#define CAFFE_NUMBER_SERIES_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class NumberSeriesDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit NumberSeriesDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~NumberSeriesDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "NumberSeriesData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  vector<int> origin_;
  int lines_id_;
};


}  // namespace caffe

#endif //CAFFE_NUMBER_SERIES_DATA_LAYER_HPP_
