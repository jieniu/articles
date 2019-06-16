# const 关键字

`const` 是 `C++` 中的关键字，**它会在编译期间（时机很重要），告诉编译器这个对象是不能被修改的**。初学者一般会认为 `const` 是个麻烦的东西，因为它常常让你的程序编译不通过，而去掉了 `const` 之后，就不会有这么多「问题」了，实不相瞒，其实我刚学 `C++` 的时候，有一段时间就处于这种状态。



但实际上，`const` 非常有用：

1. 他可以保护变量，防止它们被无意的修改。
2. 它可以传达有效的信息给读你代码的人：这个变量是不能被修改的，或者这个函数不会修该对象的成员，等。
3. `const` 关键字可以优化编译结果，让可执行程序更为紧凑。
4. `const` 修饰的变量还可以写入 ROM 介质。

总之，我认为合理的使用 `const` 可以让你写出来的程序更为严谨、健壮，不易出错，乃至于更加高效。



### const 与变量

用 `const` 修饰常量，比较容易混淆的是：「 `const` 类型的常量指针」和「指针所指向的内容为 `const` 」之间的区别：

```java
int i = 0;
const int* p1 = &i; 
int const* p2 = &i; 
```

上面代码表示指针所指的内容为 `const`，你不能做 `*p1 = 1;` 这样的操作，但可以 `p1++;`

```java
int* const p3 = &i; 
const int* const p4 = &i; 
```

以上代码中，`p3` 为 `const` 指针，你无法修改 `p3`，如 `p3++;`，但你可以执行 `*p3 = 1;`。最后一行中，`p4` 和其所指向的内容都是 `const` 的。这里的规律是：**如果 `const` 在 `*` 号左边，就表示指针所指的内容是常量，否则指针本身是常量。**



另外，我们要尽可能的减少对 `const` 常量的强制转换操作，即将 `const` 变量强制转换为非 `const` 的，或相反，尤其是前者，因为这会破坏你对常量赋予的承诺，可能会导致误操作，从而引入隐蔽 `bug`。

```java
const int i = 9;
// 强制将常量 i 转换为普通变量
const_cast<int&>(i) = 6;  
int j;
// 编译失败; 因为该语句强制将变量 j 转换成了常量
static_cast<const int&>(j) = 7; 
```



### const 与函数

`const` 与函数结合，我们需要考虑3种情况：

1. `const` 常量作为函数的参数传入
2. 函数返回 `const` 类型
3. 声明类的成员函数为 `const`



当 `const` 常量作为参数传入时，**该常量一定需要是引用类型**，否则 `const` 起不到应有的作用，例如

```java
class Dog {
    string name_;
public:
    Dog(const string& name) { name_ = name; }
	void setName(const string name);
	void setName(string name);
};
```

上面两个 `setName()` 函数等价，因为它们都是值传递（pass-by-copy），进入函数的参数是一个临时副本，既然是一个副本，修改它或不修改它对外部没有任何影响，所以这样的声明没有意义。要使上面的 `const` 产生意义，应该改为传引用（pass-by-reference）的方式

```java
	void setName(const string& name);
...
	Dog dog("");
	string strname = "xiao-D";
	dog.setName(strname)
```

此时，外部调用 `setName` 就不用担心该函数会修改实参 `strname` 了。除此之外， `setName` 还可以被重载（overload）

```java
	void setName(string& name); // 这里也只能是引用传递
	void setName(const string& name);
```

调用规则为，当你的实参为 `const` 类型时，会调用 `void setName(const string& name);` 函数，否则调用 `void setName(string& name);`



和函数的参数为 `const` 一样，当函数返回 `const` 类型时，也需要是引用类型，否则没有意义，即

```java
	const string& getName() { return name_; }
```

这样可以防止外部对 object 的内部成员进行修改。



最后，我们来看一下 `const` 类型的函数，将以下函数加到 `Dog` 类中：

```java
	void bark() const { cout << "dog " << name_ << " bark."; }
```

`const` 类型的函数表明该函数不会修改对象的成员，同时该函数也不可以调用非 `const` 函数，例如如下操作都是不允许的

```java
	void bark() const {
        name_ = "da-D"; // can't modify any member
        setName("mark"); // can't call a non-const function
	}
```

除此之外，`const` 类型的函数也是可以重载的，即

```java
	void bark() const { cout << "const dog " << name_ << " bark." << endl; }
	void bark() { cout << "non-const dog " << name_ << " bark." << endl; }
```

调用规则是，当对象是 `const` 类型时，调用 `const` 类型的函数，否则调用非 `const` 类型的函数。

```java
Dog d1;
d1.bark("QQ");
const Dog d2;
d2.bark("Wechat");
```

得到的结果为：

```bash
non-const dog QQ bark.
const dog Wechat bark.
```



参考：

* https://www.youtube.com/watch?v=7arYbAhu0aw&list=PLE28375D4AC946CC3&index=1
* https://www.youtube.com/watch?v=RC7uE_wl1Uc
