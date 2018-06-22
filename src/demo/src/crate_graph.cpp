/**********************************************
*Time:2018/04/17
*
*function:create Graph
*By : haosen
************************************************/
#include <ros/ros.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/public/session.h>


using namespace tensorflow;

GraphDef CreateGraphDef()
{
  Scope root = Scope::NewRootScope();

  auto X = ops::Placeholder(root.WithOpName("x"),DT_FLOAT,ops::Placeholder::Shape({-1,2}));

  auto A = ops::Const(root, { { 3.f, 2.f },{ -1.f, 0.f } });

  auto Y = ops::MatMul(root.WithOpName("y"),A,X,ops::MatMul::TransposeB(true));

  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));

  return def;
}


int main(int argc, char **argv)
{
  GraphDef graph_def = CreateGraphDef();

  SessionOptions options;
  std::unique_ptr<Session> session(NewSession(options));
  TF_CHECK_OK(session->Create(graph_def));


  std::vector<float> data = {1,2,3,4};
  auto mapped_X = Eigen::TensorMap<Eigen::Tensor<float,2,Eigen::RowMajor>> (&data[0],2,2);
  auto eigen_X = Eigen::Tensor<float,2,Eigen::RowMajor>(mapped_X);

  Tensor X_(DT_FLOAT,TensorShape({ 2 , 2 }));
  X_.tensor<float,2>()= eigen_X;

  std::vector<Tensor>outputs;
  TF_CHECK_OK(session->Run({ { "x", X_ } }, { "y" }, {}, &outputs));

  // Get the result and print it out
   Tensor Y_ = outputs[0];
   std::cout << Y_.tensor<float, 2>() << std::endl;

   session->Close();



  ROS_INFO("----------------end!-------------");
}
