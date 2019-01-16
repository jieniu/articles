# 为什么不要使用全局变量

在写程序时，我们都知道一条规范：不要使用全局变量。至于为什么，有可能是因为它会污染命名空间，也有可能是因为它会造成程序的不确定性，本文主要使用一个例子，来说明全局变量是如何让程序变得不确定的。

我们先定义两个类，一个 `Cat`，一个 `Dog`，如下是 `cat.h` 和 `cat.cc` 文件

```c++
// cat.h
#include <iostream>
using namespace std;

class Cat;
extern Cat c;

class Cat {
public:
    Cat(char* name);
    void meow();
private:
    char* _name;
};

// cat.cc
#include "cat.h"
#include "dog.h"

Cat c("mimi");
Cat::Cat(char* name) {
    cout << "construct cat" << endl;
    _name = name;
}

void Cat::meow() {
    cout << "cat " << _name << " meow" << endl;
}
```

`Cat` 类很简单，只有一个成员 `_name`，它是一个指针变量，且通过 `meow` 方法把它打印到屏幕上，同时，我们还定义了一个全局变量 `Cat c("mimi");`，同样的，`Dog` 类的定义也很简单，如下：

```
// dog.h
#include <iostream>
using namespace std;

class Dog;
extern Dog d;

class Dog {
public:
    Dog(char* name);
    void bark();
private:
    char* _name;
};

// dog.cc
#include "dog.h"
#include "cat.h"

Dog d("kobe");
Dog::Dog(char* name) {
    cout << "construct dog" << endl;
    _name = name;
}                                                                   
                                                                    
void Dog::bark() {                                                  
    cout << "dog " << _name << " bark" << endl;                     
}
```

我们给 `Dog` 也定义了一个全局变量 `d("kobe");`，此时，我们修改一下 `Cat` 的构造函数，在里面引用全局变量 `d`，看看会发生什么

``` 
Cat::Cat(char* name) {
    cout << "construct cat" << endl;
    _name = name;
    d.bark();
}
```

编译运行前，别忘了我们的入口文件 `main.cc`，如下

```
int main() {
    return 0;
}
```

现在，我们将其进行编译运行

```
...
g++ -o main main.o cat.o dog.o -std=c++11  // 编译时的输出，说明了链接顺序
$ ./main
construct cat
// [1]    50759 segmentation fault  ./main 
```

可以看到程序崩溃了，崩溃原因是我们刚才在 `Cat` 构造函数中增加的一行调用全局变量的代码，因为在调用这行代码时，全局变量 `d` （实际上是 `d._name` ）还没初始化。

而全局变量的初始化顺序是由编译器决定的，所以如果我们的全局变量间又有互相依赖的话，就很容易造成程序崩溃。

避免使用全局变量的方法也有很多，其中最广泛的应数 Singleton 模式了，针对上面的代码，我们可以定义一个 `Singleton` 类，其中包含 `Cat` 和 `Dog` 的静态指针，如下

```c++
// singleton.h
class Cat;
class Dog;
class Singleton {
    static Dog* pd;
    static Cat* pc;
public:
    ~Singleton();

    static Dog* getDog();
    static Cat* getCat();
};
```

与此同时，我们还声明了两个静态方法，用来获取 `Dog`  或 `Cat` 的指针，同时，我们希望 `Dog` 和 `Cat` 是以 lazy 的方式进行初始化的，即下面的 singleton.cc 文件

```
// singleton.cc
#include "singleton.h"
#include "dog.h"
#include "cat.h"

Dog* Singleton::pd = 0;
Cat* Singleton::pc = 0;
Singleton::~Singleton() {
    delete pd;
    delete pc;
    pd = 0;
    pc = 0;
}
Dog* Singleton::getDog() { 
    if (pd == 0) {
        pd = new Dog("kobe");  
    }
    return pd;
}
Cat* Singleton::getCat() {
    if (pc == 0) {
        pc = new Cat("mimi");
    }
    return pc;
}
```

可以看到初始化 `Cat` 和 `Dog` 的时机是在第一次调用 `getCat` 和 `getDog` 时。现在你就可以删掉程序中的全局变量了，当你需要使用 `Cat` 对象或 `Dog` 对象时，直接调用 `Singleton::getCat()` 或 `Singleton::getDog()` 即可。



参考：

* https://www.youtube.com/watch?v=hE77OSTE2J0&t=87s



