# 禁止成员函数（disallow functions）

本文将介绍禁止编译器自动生成某些函数的2种方法，及在某些场景下（例如嵌入式编程中），禁止析构函数给程序带来的好处。


还记得如何禁止默认构造函数吗，即定义一个带参数的构造函数即可，如下面的代码将会编译失败

```
class OpenFile {
public:
    // 定义一个带参构造函数
	OpenFile(string filename) {/*...*/}
};

int main() {
    OpenFile f; // error: no matching constructor for initialization of 'OpenFile'
}
```



除此之外，我们还可以显示的禁止某些函数的定义

1. 在 C++ 03 中，我们可以将这些函数声明在 `private` 作用域中，且不定义它们
2. 在 C++ 11 中，提供了 `delete` 关键字来实现此功能



假设你有一个文件类 `OpenFile`，你不希望这类对象互相复制，因为这会把文件写乱，此时你可以禁止该类的复制构造函数和赋值操作符，在 C++ 03 中，你可以这样做

```
class OpenFile {
private:
	OpenFile(OpenFile& rhs);
	OpenFile& operator=(const OpenFile& rhs);
}；
```

C++ 11中是这样的

```
class OpenFile {
public:
	OpenFile(OpenFile& rhs) = delete;
	OpenFile& operator=(const OpenFile& rhs) = delete;
}；
```



在某些情况下，如果你不希望继承来自基类的函数，你也可以这样显示声明

```
class Base {
public:
	void foo();
};

class Derived : public Base {
public:
	void foo() = delete; // 不继承 foo()
};

int main() {
    Derived d;
    d.foo(); // error: attempt to use a deleted function
}
```



**禁止析构函数**

在嵌入式编程中，由于栈空间比较小的原因，我们会避开将一些大对象存储在栈中，而选择将他们存放在堆中，栈中对象的特点是：当对象离开局部空间（函数或程序块），存储在栈中的对象会自动释放，对象的析构函数会被调用，此时，如果我们将对象的析构函数定义在 `private` 域中，即禁止外部释放对象，就可以有效地保护对象不被存储在栈中。

当然，存储在堆中的对象还是要提供销毁功能的，你可以额外定义一个「自定义的析构函数」，如下：

```
class BigBlock {
public:
	BigBlock();
	void destroyMe() {delete this;}
private:
	~BigBlock() {/*...*/}
};

int main() {
    BigBlock *b = new BigBlock();
    b->destroyMe(); 
}
```


总结，本文主要介绍了以下内容

1. C++ 11: f() = delete; 使用 `delete` 关键字
2. C++ 03: 将函数声明在 `private` 中，且不定义它
3. `private` 析构函数: stay out of stack.

参考：
* [Advanced C++: Disallow Functions](https://www.youtube.com/watch?v=EL30-a2gblQ&t=0s&list=PLE28375D4AC946CC3&index=6)
