#ifndef NETWORK_H
#define NETWORK_H

#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/public/session.h>
#include "tensorflow/cc/client/client_session.h"
#include <string>
#include <vector>
#include <map>
#include <typeinfo>

using namespace tensorflow;
using namespace tensorflow::ops;


class Network
{
public:
  Network(std::vector<Output> inputs,bool trainable)
  {
    inputs=inputs;
    this->trainable=true;
  }
  ~Network();
  void setup()
  {
  }
  void load(std::string data_path,Scope root,tensorflow::ops::Save saver,bool ignore_missing)
  {
    ClientSession session(root);
    std::vector<Tensor> output;
    if(data_path.substr(data_path.find_last_of(".")+1)=="ckpt")
    {
      TF_CHECK_OK(session.Run({Restore(root,-1,data_path,DT_FLOAT)},&output));
    }
    else
    {
      //
    }
  }
  std::string * feed(const std::vector<std::string> &names)
  {
    int i;
    int n = names.size();
    for(i=0;i<n;i++)
    {
      inputs.push_back(layers[names[i]]);
    }
  }
  std::string get_output(std::string layer)
  {

  }
  //(prefix,id)
  std::string get_unique_name(std::string prefix);
  std::string make_var(std::string name,tensorflow::ops::Shape shape,bool initializer,bool trainable);
  bool validate_padding(std::string padding)
  {
    return padding=="SAME"||padding=="VALID";
  }
  Output conv(int k_h,int k_w,int c_o,int s_h,int s_w,Scope root, bool relu,std::string padding,int group, bool trainable)
  {
    if(validate_padding(padding) == false)
    {
      std::cerr << "Warning: the padding parameter is neither SAME nor VALID; this parameter is set to SAME by default." << std::endl;
      padding = "SAME";
    }
    std::vector<Tensor> output_list, input_list, kernel_list;
    std::vector<Output> conv_list;
    Output input = inputs.front();
    ClientSession session(root);
    TF_CHECK_OK(session.Run({input}, &output_list));
    const int input_dims = output_list[0].dims();
    const int c_i = (int)(output_list[0].dim_size(input_dims-1));
    assert(c_i%group==0);
    assert(c_o%group==0);
    Variable kernel(root.WithOpName("weights"), {k_h, k_w, c_i/group, c_o}, DT_FLOAT);
    Variable biases(root.WithOpName("biases"), {c_o}, DT_FLOAT);
    TF_CHECK_OK(session.Run({Assign(root, kernel, Multiply(root, TruncatedNormal(root, {k_h, k_w, c_i/group, c_o}, DT_FLOAT), Const(root, 0.01f)))}, NULL));
    TF_CHECK_OK(session.Run({Assign(root, biases, Tensor(DataType::DT_FLOAT, {c_o}))}, NULL));
    if (group==1)
    {
      conv_list.push_back(Conv2D(root, input, kernel, {1, s_h, s_w, 1}, padding));
    }
    else
    {
      OutputList input_group = Split(root, Const(root, 3), input, group).output;
      OutputList kernel_group = Split(root, Const(root, 3), kernel, group).output;
      TF_CHECK_OK(session.Run(input_group, &input_list));
      TF_CHECK_OK(session.Run(kernel_group, &kernel_list));
      int i, group_size = std::min(input_list.size(), kernel_list.size());
      std::vector<Input> output_group;
      for(i=0; i<group_size; i++) output_group.push_back(Conv2D(root, input_list[i], kernel_list[i], {1, s_h, s_w, 1}, padding));
      conv_list.push_back(Concat(root, tensorflow::gtl::ArraySlice<Input>(output_group), Const(root, 3)));
    }
    Output bias = BiasAdd(root, conv_list.front(), biases);
    if (relu==true)
    {
      inputs.clear();
      inputs.push_back(Relu(root, bias));
    }
    else
    {
      inputs.clear();
      inputs.push_back(bias);
    }
  }
  Output relu(Input input,Scope root)
  {
    Relu(root,input);
  }
  Output max_pool(Input input,int k_h,int k_w,int s_h,int s_w,std::string name, std::string padding)
  {
    if(validate_padding(padding))
    {
      //maxpool
    }
  }
  Output avg_pool(Input input,int k_h,int k_w,int s_h,int s_w,std::string name, std::string padding)
  {
    if(validate_padding(padding))
    {
      //avg_pool
    }
  }
  Output roi_pool(Input input, int pooled_height,int pooled_width,int spatial_scale,std::string name)
  {

  }
  Output proposal_layer(Input input, int _feat_stride,int anchor_scales,int cfg_key,std::string name);
  Output anchor_target_layer(Input input,int _feat_stride,int anchor_scales,std::string name);
  Output proposal_target_layer(std::vector<OutputList> input,int classes, std::string name)
  {
    //if(typeid(input[0])==typeid());

  }
  //reshape_layer
  Output reshape_layer(Input input,int d,Scope root,std::string name)
  {
    auto input_shape = Shape(root,input);
    ClientSession Session(root);
    std::vector<Tensor> outputs;

    TF_CHECK_OK(Session.Run({input_shape},&outputs));

    int size_shape_0= outputs[0].dim_size(0);
    int size_shape_1= outputs[0].dim_size(1);
    int size_shape_2= outputs[0].dim_size(2);
    int size_shape_3= outputs[0].dim_size(3);

    if(name == "rpn_cls_prob_reshape")
    {
      //tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32)
      //tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],int(d),xx,input_shape[2]])
      Cast(root,Multiply(root,Div(root,Cast(root,{size_shape_1},DT_FLOAT),Cast(root,{d},DT_FLOAT)),Cast(root,{size_shape_3},DT_FLOAT)),DT_FLOAT);

      //Reshape(root,{Transpose(root,input,{0,3,1,2})},{-1,size_shape_0,});
    }
  }
  //feature_extrapolating*
  Output feature_extrapolating(Input input,int scales_base,int num_scale_base,int num_per_octave,std::string name)
  {
    return feature_extrapolating(input,scales_base,num_scale_base,num_per_octave,name);
  }
  //lrn
  Output lrn(Input input,int radius,float alpha, float beta,std::string name,float bias)
  {
    //local response normalization最早是由Krizhevsky和Hinton在关于ImageNet的论文里面使用的一种数据标准化方法
  }
  //concat
  Output concat(InputList inputs, Tensor axis,Scope root)
  {
    return Concat(root,inputs,axis);
  }
  Output fc(Input input,int num_out,std::string name,bool relu,bool trainable);
  //softmax
  Output softmax(Input input,std::string name,Scope root)
  {
    auto input_shape = tensorflow::ops::Shape(root,input);

    ClientSession Session(root);
    std::vector<Tensor> outputs;

    TF_CHECK_OK(Session.Run({input_shape},&outputs));

    int size_shape_1= outputs[0].dim_size(1);
    int size_shape_2= outputs[0].dim_size(2);
    int size_shape_3= outputs[0].dim_size(3);
    if(name=="rpn_cls_prob")
    {
      //Reshape(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor,::tensorflow::Input shape);
      auto reshape = Reshape(root,input,{-1,size_shape_3});
      return Reshape(root,Softmax(root,reshape),{-1,size_shape_1,size_shape_2,size_shape_3});
    }
    else
    {
      return Softmax(root,input);
    }
  }
  //dropout*
  Output dropout(Input input,Tensor keep_prob,std::string name,Scope root)
  {
    //return dropout(input,)
  }





  std::vector <Output> inputs;
  std::map<std::string, Output> layers;
  bool trainable;



};

#endif // NETWORK_H
