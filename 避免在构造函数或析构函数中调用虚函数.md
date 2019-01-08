# 避免在构造函数或析构函数中调用虚函数

我们先来看下下面程序

```c++
#include <iostream>
using namespace std;
                                            
class Dog {
public:
    Dog () { cout << "I'm a dog" << endl; }
    virtual ~Dog () { cout << "destroy a dog" << endl; }

	void bark() { cout << "dog bark" << endl; }
    void seeDog() { bark(); }
};

class YellowDog : public Dog {
public:
    YellowDog() { cout << "I'm a yellow dog" << endl; }
    virtual ~YellowDog() { cout << "destroy a yellow dog" << endl;}

    void bark() { cout << "yellow dog bark" << endl; }
};


int main() {
    YellowDog yd = YellowDog();
    yd.seeDog();

    return 0;
}
```

运行上述程序后，得到以下输出：

```
I'm a dog
I'm a yellow dog
dog bark
destroy a yellow dog
destroy a dog
```

可以看到在创建 `YellowDog` 对象时，首先会调用基类的构造函数，而且虽然是 `YellowDog` 调用的 `seeDog`，但执行的却是 `Dog::bark`，此时，如果你想要调用 `YellowDog::bark`，你需要把基类 `Dog` 中的 `bark` 声明为 `virtual` 类型，如下：

```C++
	virtual void bark() { cout << "dog bark" << endl; }
```

同时，为了更好的可读性，建议你也将子类中的继承自基类的虚函数也显示的声明为 `virtual`，此时你再运行上面的程序后，就可以看到 `YellowDog::bark` 被调用了，这就是 C++ 中的多态或动态绑定。

```
I'm a dog
I'm a yellow dog
yellow dog bark
destroy a yellow dog
destroy a dog
```

下面，我们做点小改动，故意让 C++ 中的动态绑定失效，我们把 `seeDog()` 中的 `bark()` 挪到 `Dog` 的构造函数中，看下会出现什么情况

```c++
	// ...
    Dog () { cout << "I'm a dog" << endl; bark(); }
	// ...
```

此时输出变成了

```
I'm a dog
dog bark
I'm a yellow dog
destroy a yellow dog
destroy a dog
```

可以看到虽然声明了虚函数，但却没有进行动态绑定，原因在于**在调用 `Dog` 的构造函数时，`YellowDog` 对象还没有被构造完成，对一个不存在的对象调用其成员函数是非常危险的**，所以编译器在这里选择了调用 `Dog::bark`，而不是 `YellowDog::bark`，同时要注意的是，**在构造函数中尽可能的只做最简单的初始化操作，避免复杂的函数调用。**

现在我们把 `bark()` 放在 `Dog` 中的析构函数中，看下会发生什么情况

```c++
    // ...
	Dog () { cout << "I'm a dog" << endl;  }
    virtual ~Dog () { cout << "destroy a dog" << endl; bark();}
	// ...
```

输出如下

```bash
I'm a dog
I'm a yellow dog
destroy a yellow dog
destroy a dog
dog bark
```

再一次，多态的特性没有生效，原因在于在析构 `YellowDog` 的对象时，先调用 `YellowDog` 的析构函数，即在调用 `bark` 时，`YellowDog` 部分的数据已经被清理，此时再调用该对象的虚函数也是非常危险的，所以为了安全起见，编译器又一次选择了调用基类的 `bark`。

以上，我们学到了一条宝贵的经验：

> 不要在构造函数或析构函数中调用虚函数



参考：https://www.youtube.com/watch?v=mTE5jaXaOuE&t=0s&index=9&list=PLE28375D4AC946CC3