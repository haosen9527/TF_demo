#include <vector>
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/graph/default_device.h"
#include <opencv2/core/core.hpp>



using namespace tensorflow;

GraphDef CreateGraphDef()
{
  Scope root = Scope::NewRootScope();

  Status status ;

  // create tensorflow structure start
  auto weights = ops::Variable(root,{},DT_FLOAT);
  auto assign_W = tensorflow::ops::Assign(root, weights, tensorflow::ops::RandomNormal(root, {.3}, DT_FLOAT));

  auto biases = ops::Variable(root, {},DT_FLOAT);
  auto assign_b = tensorflow::ops::Assign(root, biases, tensorflow::ops::RandomNormal(root, {.3}, DT_FLOAT));

  auto x = ops::Placeholder(root,DT_FLOAT);

  auto Y = tensorflow::ops::Add(root,biases,tensorflow::ops::Multiply(root,weights,x));//weights*x_data+biases;

  //auto loss = ops::ReduceMean(root,Square(Y-y_data),{0.0});

  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));

  return def;
}


int main(int argc, char **argv)
{
  //init graph
//  GraphDef graph_def = CreateGraphDef();

  Session* session;
  Status status = NewSession(SessionOptions(), &session);

  Scope root = Scope::NewRootScope();

  // create tensorflow structure start
  auto weights = ops::Variable(root,{},DT_FLOAT);
  auto assign_W = tensorflow::ops::Assign(root.WithOpName("W"), weights, tensorflow::ops::RandomNormal(root, {.3}, DT_FLOAT));

  auto biases = ops::Variable(root, {},DT_FLOAT);
  auto assign_b = tensorflow::ops::Assign(root.WithOpName("b"), biases, tensorflow::ops::RandomNormal(root, {.3}, DT_FLOAT));

  auto x = ops::Placeholder(root.WithOpName("x"),DT_FLOAT);

  auto Y = tensorflow::ops::Add(root.WithOpName("y"),biases,tensorflow::ops::Multiply(root,weights,x));//weights*x_data+biases;


  GraphDef graph_def;
  root.ToGraphDef(&graph_def);
  session->Create(graph_def);

  //tensor input
  Tensor a(DT_FLOAT,TensorShape());
  a.scalar<float>()() = 4;
  Tensor b(DT_FLOAT,TensorShape());
  b.scalar<float>()() = 2;
  Tensor c(DT_FLOAT,TensorShape());
  c.scalar<float>()() = 10;

  std::vector<std::pair<std::string,tensorflow::Tensor>> inputs = {
    {"W",a},
    {"b",b},
    {"x",c},
  };

  std::vector<Tensor>outputs;
  session->Run(inputs,{"y"},{}, &outputs);

  //auto out_y = outputs[0].scalar<float>();

  std::cout <<outputs[0].DebugString()<<std::endl;
//  std::cout <<out_y<<std::endl;
  session->Close();

  std::cout<<"----------------end!-------------"<<std::endl;
}
