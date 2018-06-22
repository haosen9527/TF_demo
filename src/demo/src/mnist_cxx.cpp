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

#include "/home/micros/catkin_new/src/demo/include/demo/data_set.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

int main() {
  string path = "/home/micros/tensorflow/data/";
  DataSet train_data(path, "train-images.idx3-ubyte","train-labels.idx1-ubyte");
  DataSet  test_data(path, "t10k-images.idx3-ubyte" ,"t10k-labels.idx1-ubyte");


  Scope root = Scope::NewRootScope();
  ClientSession session(root);
//构造训练数据,
  Tensor train_X(DT_FLOAT, TensorShape({60000,784}));
  Tensor train_Y(DT_FLOAT, TensorShape({60000,10}));

  copy_n(train_data.images().begin(), train_data.images().size(),
         train_X.flat<float>().data());
      //cout<< "test: "<< train_X.Slice(0,1).flat<float>()<<endl;

  copy_n(train_data.labels().begin(), train_data.labels().size(),
         train_Y.flat<float>().data());
      cout<< train_Y.Slice(0,1).flat<float>()<<endl;


//构造测试数据,
  Tensor test_X(DT_FLOAT, TensorShape({10000,784}));
  Tensor test_Y(DT_FLOAT, TensorShape({10000,10}));

  copy_n(test_data.images().begin(), test_data.images().size(),
         test_X.flat<float>().data());
      //cout<< test_X.matrix<float>().data()[10000]<<endl;

  copy_n(test_data.labels().begin(), test_data.labels().size(),
         test_Y.flat<float>().data());



//构建训练模型 Placeholder(); Variable(); Softmax();
  //定义输入
  auto X = Placeholder(root,DT_FLOAT);

  //定义模型参数
  auto W = Variable(root,{784,10},DT_FLOAT);
  auto b = Variable(root,{10},DT_FLOAT);
  //模型参数初始化
  auto assign_W = Assign(root, W, RandomNormal(root, {784,10}, DT_FLOAT)); //手动给变量赋初值
  auto assign_b = Assign(root, b, RandomNormal(root, {10}, DT_FLOAT));

  //定义输出
  auto y = Softmax(root, Add(root, MatMul(root, X, W), b)); //模型数据输出
  auto Y= Placeholder(root,DT_FLOAT); //训练数据输出

  //定义模型评估函数
  auto Hy = Mul(root, Y, Log(root, y)) ;
  auto cross_entropy = Mul(root, -1.f, ReduceSum(root, Hy, {0, 1})); //计算矩阵各项之和; 0:行向量相加; 1:列向量相加; {0, 1}:全部元素相加
  //      vector<Tensor> outputs;
  //      TF_CHECK_OK(session.Run({cross_entropy}, &outputs));
  //      cout<<outputs[0].flat<float>()<<endl;

  // add the gradients operations to the graph
  vector<Output> grad_outputs;
  TF_CHECK_OK(AddSymbolicGradients(root, {cross_entropy}, {W, b}, &grad_outputs)); //向计算图中增加梯度标记

  // update the weights and bias using gradient descent //输入变量w1,下降速率alpha,变量更新值delta,输出更新后的变量w1 -= alpha*delta
  auto apply_W = ApplyGradientDescent(root, W, Cast(root, 0.012,  DT_FLOAT), {grad_outputs[0]});
  auto apply_b = ApplyGradientDescent(root, b, Cast(root, 0.012,  DT_FLOAT), {grad_outputs[1]});

  //prediction , lable为one_hot编码,故用ArgMax(1:取列)还原手写数字的值,返回index; 并用 equal 逐位计算是否正确
  auto correct_prediction = Equal(root, ArgMax(root, y, 1), ArgMax(root, Y, 1));
  //bool -> float, Mean 计算平均值: sum_n/n 即为正确概率
  auto accuracy = ReduceMean(root,Cast(root, correct_prediction, DT_FLOAT),{0});


//初始化session
  //ClientSession session(root);
  vector<Tensor> outputs;

  //模型参数,TF变量初始化
  TF_CHECK_OK(session.Run({assign_W, assign_b}, nullptr));

  // training steps 训练迭代次数, 学习率等参数都可调,可以尝试修改进行测试
  cout << "start training steps ....." <<endl;
  for (int i = 0; i < 60000; i=i+20) {
      TF_CHECK_OK(session.Run({{X, train_X.Slice(i,i+20)}, {Y, train_Y.Slice(i,i+20)}},
                  {cross_entropy, W, b, y}, &outputs));
      if (i % 1000 == 0) {
          cout << "Loss after " << i << " steps :"
               << " cross_entropy = "  << outputs[0].flat<float>()
               << endl;
            cout << "accuracy:  " << outputs[0].scalar<float>() << endl;
      }
      // nullptr because the output from the run is useless
      TF_CHECK_OK(session.Run({{X, train_X.Slice(i,i+20)}, {Y, train_Y.Slice(i,i+20)}}, {apply_W, apply_b}, nullptr));
  }

  // prediction
  cout << "start predicting steps ....." <<endl;
  TF_CHECK_OK(session.Run({{X, test_X}, {Y, test_Y}}, {accuracy}, &outputs));
  cout << "accuracy:  " << outputs[0].scalar<float>() << endl;
  return 0;

}//END MAIN
