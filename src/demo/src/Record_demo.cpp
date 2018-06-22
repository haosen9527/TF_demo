/*********************************
* Time : 2018/04/18
* Function : Record/image(jpg/png)/file/
* By : haosen
***********************************/

#include <tensorflow/core/public/session.h>
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/core/lib/core/stringpiece.h"

//using namespace tensorflow;
//using namespace ops;

int main(int argc, char **argv)
{ 
  tensorflow::StringPiece data;
  tensorflow::Scope root= tensorflow::Scope::NewRootScope();
  tensorflow::int64 num;
  //从文件输出固定长度记录的阅读器。
  tensorflow::ops::FixedLengthRecordReader x(root,num);

  tensorflow::ops::TFRecordReader TF_reader(root);

}
