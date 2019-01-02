# 虚析构函数和智能指针

## 虚析构函数

当我们想在程序中实现多态时，我们经常会这样做：

```C++
#include <iostream>
using namespace std;
class Dog {
public:
  Dog() {}
  ~Dog() { cout << "Dog destroy\n";}
};

class YellowDog : public Dog {
public:
  YellowDog() {}
  ~YellowDog() { cout << "YellowDog destroy\n";}
  static Dog* createDog() { return new YellowDog(); }
};
```

即用工厂类或工厂方法来创建具体的对象，而在运行时通过对基类（这里是Dog）的指针或引用来实现对不同子类（这里是YellowDog）的调用，这样我们就实现了「多态」。不过，上面代码是有问题的，你可以看下面代码的输出：

```c++
int main(int argc, char** argv) {
  Dog* dog = YellowDog::createDog();
  delete dog;
  return 0;
}
----<output>----
Dog destroy
```

从输出结果可以看出，`delete dog` 时，只调用了基类的析构函数，而子类对象没有被析构，此时很可能会发生内存泄露，为了避免这种情况，我们需要在基类析构函数前加上 `virtual` 关键字，如下

```C++
virtual ~Dog() { cout << "Dog destroy\n"; }
```

接着重新编译后再运行程序，便可以看到子类对象被如期析构了

```
----<output>----
YellowDog destroy
Dog destroy
```

## 用智能指针 shared_ptr 实现动态析构

从 C++ 11 起，通过 `shared_ptr` 你同样可以实现动态析构，`shared_ptr` 定义在头文件 `<memory>` 中

```c++
template< class T > class shared_ptr;
```

下面我们就来看下智能指针的版本

```c++
class Dog {
public:
  Dog(){}
  ~Dog() { cout << "Dog destructor" << endl; }
};

class YellowDog : public Dog {
public:
  YellowDog() {}
  ~YellowDog() { cout << "YellowDog destructor" << endl; }

  static shared_ptr<Dog> createDog() {
      return shared_ptr<YellowDog>(new YellowDog());
  }
};

int main() {
    shared_ptr<Dog> p = YellowDog::createDog();
    return 0;
}
```

从输出中可以看到虽然 `Dog` 类中的析构函数没有声明 `virtual`，但 `YellowDog` 的析构函数仍然被顺利调用了。

_注意：`STL` 中的所有类都没有 `virtual` 析构函数，所以当你从 `STL` 中派生子类时，要尽可能的使用 `shared_ptr`_。



参考：https://www.youtube.com/watch?v=ZiNGWHg5Z-o&index=6&list=PLE28375D4AC946CC3