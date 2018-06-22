#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"

#include <tensorflow/core/public/session_options.h>
#include "tensorflow/cc/ops/io_ops.h"
#include <tensorflow/cc/framework/scope.h>

#include <vector>
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include <math.h>

#include <iostream>

using namespace std;
using namespace tensorflow;
using namespace ops;

int main()
{
    Session* session;
    Status status = NewSession(SessionOptions(), &session);

    int num=1;

    Scope root=Scope::NewRootScope();
    auto c1 =Const(root,{ {1,1} });//"tensorflow/core/public/session.h"
    auto m = MatMul(root,c1,{{41},{1}});//"tensorflow/core/public/session.h"

    FixedLengthRecordReader(root,num);


    GraphDef gdef;
    Status s = root.ToGraphDef(&gdef);

    if(!s.ok())
    {
      cout<<"-------------"<<endl;
    }

    s.Update(s);

    if (!status.ok()) {
        cout << status.ToString() << "\n";
        return 1;
    }
    cout << "Session successfully created.\n";
}
