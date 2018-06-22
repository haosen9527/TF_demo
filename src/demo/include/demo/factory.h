#ifndef FACTORY_H
#define FACTORY_H

/********************************************
 *Factory method for easily getting imdbs by name.
 *by:lvjianghao
 *******************************************/
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/public/session.h>
#include <vector>
#include <string>
//#include "/home/micros/catkin_new/src/demo/include/demo/factory.h"

//Get a network by name.
void get_network(std::string name)
{
  if(name.find_last_of("_test"))
  {
    //networks.VGGnet_test()
  }
  else if(name.find_last_of("_train"))
  {
    //networks.VGGnet_train()
  }
  else
  {
    std::cout<<"factory name error";
  }
}


//List all registered imdbs.
void list_network()
{
  //python 字典(Dictionary) keys() 函数以列表返回一个字典所有的键。
  //__sets.keys();

}

//std::vector<std::string> split(std::string str,std::string separator){
//    std::vector<std::string> result;
//    int cutAt;
//    while((cutAt = str.find_first_of(separator))!=str.npos){
//        if(cutAt>0){
//            result.push_back(str.substr(0,cutAt));
//        }
//        str=str.substr(cutAt+1);
//    }
//    if(str.length()>0){
//        result.push_back(str);
//    }
//    return result;
//}
#endif // FACTORY_H
