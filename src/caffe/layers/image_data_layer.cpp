#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos;
  int label;
  n_images_ = 0;
  while (std::getline(infile, line)) {
    n_images_++;
    pos = line.find_last_of(' ');
    label = atoi(line.substr(pos + 1).c_str());
    lines_.push_back(std::make_pair(line.substr(0, pos), label));
  }

  CHECK(!lines_.empty()) << "File is empty";

  /* Added by TDteach
   * Uesd to change the origin image into meanpose sense
   */
  int in_state;
  shift_scale_ = this->layer_param_.image_data_param().shift_scale();
  n_landmarks_ = this->layer_param_.image_data_param().n_landmarks();
  if (n_landmarks_ > 0) {
    string lkfile = this->layer_param_.image_data_param().landmarks();
    string mpfile = this->layer_param_.image_data_param().meanpose();

    meanpose_.resize(n_landmarks_);
    FILE *fid;
    fid = fopen(mpfile.c_str(),"r");
    in_state = fscanf(fid,"%d%d", &mm_width_, &mm_height_);
    for (int i = 0; i < n_landmarks_; i++) { 
      in_state = fscanf(fid,"%f%f", &meanpose_[i].x, &meanpose_[i].y);
    }
    fclose(fid);

    fid = fopen(lkfile.c_str(),"r");
    for (int i = 0; i < n_images_; i++) {
      FacialPose tmp;
      tmp.resize(n_landmarks_);
      for (int j = 0; j < n_landmarks_; j++) {
        in_state = fscanf(fid,"%f%f", &tmp[j].x, &tmp[j].y);
      }
      landmarks_.push_back(tmp);
    }
    fclose(fid);
  }
  CHECK_GT(in_state, 0);

  src_imgs_.clear();
  dst_imgs_.clear();
  linear_data_ratio_ = this->layer_param_.image_data_param().linear_data_ratio();
  if (linear_data_ratio_ > 0) {
    linear_src_label_ = this->layer_param_.image_data_param().linear_src_label();
    linear_dst_label_ = this->layer_param_.image_data_param().linear_dst_label();

    for (int i = 0; i < n_images_; i++) {
      if (lines_[i].second == linear_src_label_ || lines_[i].second == linear_dst_label_) {
        cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
            0, 0, is_color);
        CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
        if (n_landmarks_ > 0) {
          cv::Mat trans;
          TDTOOLS::calcTransMat(landmarks_[lines_id_], meanpose_, trans);
          TDTOOLS::cropImg(cv_img, mm_height_, mm_width_, trans);
        }
        if (new_width > 0 && new_height > 0) {
          cv::resize(cv_img, cv_img, cv::Size(new_width, new_height));
        }

        if (lines_[i].second == linear_src_label_) {
          src_imgs_.push_back(cv_img);
        }
        if (lines_[i].second == linear_dst_label_) {
          dst_imgs_.push_back(cv_img);
        }
      }
    }
  }


  //===============split line=====================================


  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    if (n_landmarks_ > 0) {
      prefetch_rng_ldmk_.reset(new Caffe::RNG(prefetch_rng_seed));
    }
    ShuffleImages();
  } else {
    if (this->phase_ == TRAIN && Caffe::solver_rank() > 0 &&
        this->layer_param_.image_data_param().rand_skip() == 0) {
      LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
    }
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
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
  label_size_ = this->layer_param_.image_data_param().label_size();
  vector<int> label_shape(2, batch_size);
  label_shape[1] = label_size_;
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
  if (n_landmarks_ > 0) {
    caffe::rng_t* prefetch_rng_ldmk =
        static_cast<caffe::rng_t*>(prefetch_rng_ldmk_->generator());
    shuffle(landmarks_.begin(), landmarks_.end(), prefetch_rng_ldmk);
  }
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  vector<int> label_shape(2,batch_size);
  label_shape[1] = label_size_;
  batch->label_.Reshape(label_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);

    cv::Mat cv_img;

    if ((rand()%10000/10000.0) < linear_data_ratio_) {
      int u = rand()%src_imgs_.size();
      int v = rand()%dst_imgs_.size();
      float rt = rand()%10000/10000.0;

      cv::addWeighted(src_imgs_[u], (1-rt), dst_imgs_[v], rt, 0, cv_img);

      //set label
      if (label_size_ > 1) {
        caffe_set(label_size_, Dtype(0), prefetch_label+item_id*label_size_);
        prefetch_label[item_id*label_size_ + linear_src_label_] = Dtype(1-rt);
        prefetch_label[item_id*label_size_ + linear_dst_label_] = Dtype(rt);
      }
    }
    else {
      /* modified by TDteach
      cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
          new_height, new_width, is_color);
      CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
      */
      cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
          0, 0, is_color);
      CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
      if (n_landmarks_ > 0) {
        cv::Mat trans;
        TDTOOLS::calcTransMat(landmarks_[lines_id_], meanpose_, trans);
        /* apply shift transformation
         */
        float shift_x = (rand()%1000/500.0-1)*shift_scale_;
        float shift_y = (rand()%1000/500.0-1)*shift_scale_;
        trans.at<float>(0,2) += shift_x*mm_width_;
        trans.at<float>(1,2) += shift_y*mm_height_;
  
        TDTOOLS::cropImg(cv_img, mm_height_, mm_width_, trans);
      }
      if (new_width > 0 && new_height > 0) {
        cv::resize(cv_img, cv_img, cv::Size(new_width, new_height));
      }

      //set label
      if (label_size_ > 1) {
        caffe_set(label_size_, Dtype(0), prefetch_label+item_id*label_size_);
        prefetch_label[item_id*label_size_ + lines_[lines_id_].second] = Dtype(1);
      }
    }

    /*debug info image
     */
    //char bb[100];
    //sprintf(bb,"/home/tangdi/tmp/%d.png",lines_id_);
    //string fn(bb);
    //cv::imwrite(fn, cv_img);
    //=======================split line====================================

    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    if (label_size_ == 1)
      prefetch_label[item_id] = lines_[lines_id_].second;

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageDataLayer);
REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe
#endif  // USE_OPENCV
