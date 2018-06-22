#ifndef VGGNET_TEST_H
#define VGGNET_TEST_H

#include "/home/micros/catkin_new/src/demo/include/demo/network.h"
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/public/session.h>
#include <string>
#include <vector>


class VGGnet_test:public Network
{
public:
  VGGnet_test(Scope root,std::vector <Output> inputs,bool trainable,std::map<std::string, Output> layers):Network(inputs,trainable)
  {
    auto data = tensorflow::ops::Placeholder(root,DT_FLOAT);
    auto im_info = tensorflow::ops::Placeholder(root,DT_FLOAT);
    auto keep_prob = tensorflow::ops::Placeholder(root,DT_FLOAT);

//    layers.insert(pair<std::string,Output>("data",data));
//    layers.insert(pair<std::string,Output>("im_info",im_info));

    layers["data"] = data;
    layers["im_info"] = im_info;

  }
  ~VGGnet_test();

  void setup()
  {
    std::vector<std::string> names;
    feed(names);
  }


public:
  std::vector <Output> inputs;
  std::map<std::string, Output> layers;
  bool trainable;




};

#endif // VGGNET_TEST_H
