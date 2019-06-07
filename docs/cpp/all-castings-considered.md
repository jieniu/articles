# 显示类型转换（Casting)

C++ 中定义了 4 种显示类型转换，初学 C++，难免觉得这一部分内容复杂、难以理解（当然我也不例外），但掌握它又是很有必要的，毕竟事物的存在，必有它存在的道理，而这个道理，就是相比其他设计而言（例如传统的 C 风格的类型转换），**C++ 的类型转换能够减少出错的概率**，我们在前面的文章中也提到过这个设计原则，想成为一名优秀的程序员，该原则值得你谨记在心。

这 4 种类型转换分别是：

1. static_cast
2. dynamic_cast
3. const_cast
4. reinterpret_cast

## static_cast

`static_cast` 既可以在对象间进行类型转换（前提是有相应的转换构造函数或转换操作符支持），也可以在指针或引用间做类型转换（一般是父类和子类之间的向上或向下转换），表达式如下

```
static_cast<new_type>(expression);
```

下面是一个简单的例子，将 `float` 类型转换为 `int` 类型

```java
#include <iostream>
using namespace std;

int main() {
  float f = 3.5;
  int i1 = f;   // C 语言的用法
  int i2 = static_cast<int>(f);
  cout << i1 << endl;
  cout << i2 << endl;
  return 0;
}
```

稍微修改上面的代码，看看如果类型转换针对的是指针类型，会发生什么

```java
#include <iostream>
using namespace std;

int main() {
  char c = 'a';
  int* i1 = (int*)&c;		// C 风格，compile ok
  int* i2 = static_cast<int*>(&c); // compile error
  cout << *i1 << endl;
  return 0;
}
```

可以看到，在 C++ 中，编译器不允许你将一个 `char*` 类型的指针转换为 `int*` 类型的，因为这样做可能会产生整型溢出的错误，虽然你依然可以采用 C 风格的强制类型转换达到这一目的，但不建议你这么做，**这也是在 C++ 中，为什么要用 `_cast` 来取代 C 类型转换的原因之一**。

但 `void*` 类型的指针是可以转换为任何其他类型的指针的，这一点也是 C 中常用的技巧：

```java
int score = 90;
void* p = &score;
int* pscore = static_cast<int*>(p);
```

此外，`static_cast` 很适合将隐式类型转换显示化，通常，使用显示类型转换是较好的编程习惯。依然以上一篇文章的例子为例：

```java
class dog {
public:
  dog(string name) {m_name = name;}           // 转换构造函数
  operator string () const { return m_name; } // 转换操作符
private:
  string m_name;
};

int main() {
  string dogname = "dog";
  dog d1 = dogname;                             // a1. 隐式 string->dog
  dog d2 = static_cast<dog>(dogname);           // a2. 显示 string->dog
  string other_dog_name = d2;                   // b1. 隐式 dog->string
  string my_dog_name = static_cast<string>(d2); // b2. 显示 dog->string
  cout << my_dog_name << endl;
  cout << other_dog_name << endl;
  return 0;
};
```

上面的代码中，注释 `a1` 和 `b1` 分别是利用隐式构造函数和隐式操作符实现的隐式类型转换，而注释 `a2` 和 `b2` 将它们显示化了 。

前文提到了 `static_cast` 还可以在父子类之间进行向上或向下的类型转换，要注意的是，子类的继承作用域需要是 `public`，且转换的对象需要是指针或引用类型

```java
#include <vector>
#include <iostream>
 
class B {
  public:
    void hello() const {
        std::cout << "Hello world, this is B!\n";
    }
};
class D : public B {
  public:
    void hello() const {
        std::cout << "Hello world, this is D!\n";
    }
};
int main()
{
  D d;
  B& br = d; // 通过隐式转换向上转型
  br.hello();
  D& another_d = static_cast<D&>(br); // 向下转型
  another_d.hello();
}
```

以上代码会输出

```
Hello world, this is B!
Hello world, this is D!
```

## dynamic_cast

`dynamic_cast` 主要针对的是有继承关系的类的指针和引用，且要求类中必须要有 `virtual` 关键字，这样才能提供**运行时类型检查**，当类型检查发现不匹配时，`dynamic_cast` 返回  0；实际运用中，`dynamic_cast` 主要用于向下转型。在 [cppreference.com](https://zh.cppreference.com/w/cpp/language/dynamic_cast) 页面，提供了很好的例子

```java
#include <iostream>
struct Base {
    virtual ~Base() {}
}; 
struct Derived: Base {
    virtual void name() {}
};
int main() {
    Base* b1 = new Base;
    if(Derived* d = dynamic_cast<Derived*>(b1)) {
        std::cout << "downcast from b1 to d successful\n";
        d->name(); // 调用安全
    } 
    Base* b2 = new Derived; // 向上转型，可以用 dynamic_cast ，但不必须
    if(Derived* d = dynamic_cast<Derived*>(b2)) {
        std::cout << "downcast from b2 to d successful\n";
        d->name(); // 调用安全
    }
    delete b1;
    delete b2;
}
```

上面代码仅输出

```
downcast from b2 to d successful
```

另一个 `cout` 没有输出的原因是 `Base* b1` 和 `Derived* d` 类型不一致，转型失败，`dynamic_cast` 返回为 0。

同样是运行时多态，使用 C++ 的虚函数列表要比 `dynamic_cast` 在性能上有很大的优势，而且前者能够写出更通用的代码（更少的类型相关的硬编码）。

## const_cast

`const_cast` 在用法上要简单很多，它能够移除 `const 引用` 或 `const 指针` 上的 `const` 属性，需要注意的是

1. 被 `const引用` 或 `const指针` 指向的变量不能也是 `const` 类型
2. `const_cast` 不能作用于函数指针或成员函数指针上

典型的使用方法如下：

```java
int i = 3;
const int& cri = i;
const int* cpi = &i;
const_cast<int&>(cri) = 4;   // i = 4
*const_cast<int*>(cpi) = 5; // i = 5
```

上面的 `i` 不可以是 `const` 类型，否则程序行为是未定义的，如

```java
const i = 3;
const int& cri = i;
const_cast<int&>(cri) = 4;   // 可编译，但行为未定义
```

`const_cast` 的典型应用场景在 logic constness，详情请参见[另一篇文章](/cpp/logic_constness_and_bitwise_constness.md)。

## reinterpret_cast

`reinterpret_cast` 是 C++ 中最危险的转型操作，它可以将指针所指的类型重新解释为其他任意类型，且没有任何检验机制。`reinterpret_cast` 主要用于这样的场景：先将一个指针转型为另一种类型（通常为 `void*`），在使用前，再把它转型为原始类型，如下：

```java
int* a = new int();
void* b = reinterpret_cast<void*>(a);
int* c = reinterpret_cast<int*>(b);
```

但实际上，这样的使用方法完全可以用 `static_cast` 代替，即

```java
int* a = new int();
void* b = static_cast<void*>(a);
int* c = static_cast<int*>(b);
```

所以，如果不能保证一定安全，尽量避免使用这种转型。

## 总结

C++ 除了支持以上 4 种转型方式，还兼容 C 风格的转型，但读了本文后，你可以把 C 风格的转型看成是 `static_cast` 、`const_cast`  和 `reinterpret_cast` 这 3 种转型的并集。所以，我们更应该使用 C++

风格的转型，而不是 C 风格的，原因是：

1. 在做 coding review 时，你可以通过搜索 `_cast` 关键字，来查看代码中哪些地方用到了转型操作
2. 每种 C++ 转型都有其使用限制，而 C 转型却没有
3. C++ 的转型具备运行时识别能力



参考：

* https://www.youtube.com/watch?v=UfrR1nNfoeY&list=PLE28375D4AC946CC3&index=17

* https://zh.cppreference.com/w/cpp/language/dynamic_cast

* https://zh.cppreference.com/w/cpp/language/static_cast

* https://www.geeksforgeeks.org/static_cast-in-c-type-casting-operators/

* https://zh.cppreference.com/w/cpp/language/const_cast

* https://stackoverflow.com/questions/19554841/how-to-use-const-cast