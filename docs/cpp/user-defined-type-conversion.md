# 用户自定义的隐式类型转换

C++ 中的类型转换包含内建类型的转换和用户自定义类型的转换，而这两者都又可分为隐式转换和显示转换，所以一共可以分如下四象限表格中的 A、B、C、D 四种情况

|                                                  | 隐式转换 | 显示转换<br>(casting) |
| ------------------------------------------------ | -------- | --------------------- |
| 内建类型转换 <br>(int, float ...)                | A        | B                     |
| 用户自定义类型转换<br>(类 vs 类; 类 vs 内建类型) | C        | D                     |

本篇只讨论隐式转换，内建类型的隐式转换举例如下

```C++
char c = 'A';
int i = c;                 // 将 char 隐式转换为 int，又称 Integral promotion
char* pc = 0;              // int 转换为 Null 指针
dog* pd = new yellowdog(); // 指针类型转换，子类 yellowdog 指针转换为父类 dog 指针
```



而用户自定义的隐式转换是重头戏，一般我们说自定义隐式转换，指两方面的转换

1. 利用接受单个参数的构造函数，可以将其他类型的对象转换为本对象
2. 使用类型转换函数， 将本类的对象转换为其他对象

两者合起来可以构成一个双向转换关系，下面我们看一个例子

```c++
class dog {
public:
	dog(string name) {m_name = name;} 
	string getName() {return m_name;}
private:
	string m_name;
};

int main() {
    string dogname = "dog";
    dog d = dogname;
	cout << "my name is " << d.getName() << \n";
    return 0;
};
```

上面例子中，`dog(string name) {m_name = name;}` 有两层含义，除了构造函数外，它还可以作为隐式转换函数，将 `string` 对象转换为 `dog` 对象，可以看到我们把 `dogname` 赋给了 `dog d`，像这样的赋值，通常是无意的行为，而且它触犯了  **is-a** 原则。

如果你不想该构造函数具备隐式转换的特性，你应该使用 `explicit` 对该函数进行声明：

```c++
explicit dog(string) {m_name = name;}
```

反过来，我们还可以定义一个转换函数，将 `dog` 对象转换为 `string`，如下

```c++
class dog {
    // ...
    operator string () const { return m_name; }
};
```

这样，下面的输出可以简化为：

```c++
cout << "my name is " << (string)d << \n";
```

可以看到，自定义类型的隐式转换很容易写出和本意不符的代码，这些代码往往容易出错，很明显，这样的设计算不上是好设计，相反，我们更应遵循这样的设计原则：

> **当我们定义一个 api 接口，我们希望正确的使用这个 api 是一件很容易的事情，而很难用错（理想情况下，当你错误使用一个 api 时，它不能被编译通过），而定义过多的转换函数，却很容易让我们的api更容易出错**

对于隐式类型转换，应该记住 2 点

1. 避免定义看上去不符合预期的转换，例如将 string 转换为一个 dog
2. 避免定义双向的类型转换，A 可以向 B 转换，但继续实现 B 向 A 的转换就过犹不及

下面我们来举一个正面的隐式转换的例子，当你的类是一个处理数字的类型时，例如：有理数类，用隐式转换比较合适：

```c++
class Rational {
public:
	Rational(int numerator = 0, int denominator = 1)
		: num(numberator), den(denominator) {}
	int num;
	int den;
};
Rational operator*(Rational lhs, Rational rhs) {
	return Rational(lhs.num*rhs.num, lhs.den*rhs.den);
}
int main() {
    Rational r1 = 23;
    Rational r2 = r1 * 2;
    Rational r3 = 3 * r1;
}
```

上面代码定义了一个有理数类 `Rational`，它的构造函数接受 2 个默认参数，分别代表分子和分母，给该构造函数传递一个参数时，`Rational` 具有隐式转换的特性，所以我们可以 `Rational r1 = 23;` 进行赋值；

为了避免双向转换，这里并没有定义将 `Rational` 转换为 `int` 的转换函数，而当我们要实现 `Rational` 对象和 `int` 之间自由的算术运算时，我们需要定义全局的操作符重载，如上面 `operator*` 定义了有理数的乘法云算符。



参考：https://www.youtube.com/watch?v=S7MNPX73FSQ&list=PLE28375D4AC946CC3&index=16&t=0s