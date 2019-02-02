# Struct 和 Class 的约定

为了降低系统的复杂性，我们需要遵循按约定编程的原则（Coding by convention），旨在：

> 减少软件开发人员需做决定的数量，获得简单的好处，而又不失灵活性。

C++ 编程也有不少约定，对于 struct 和 class 来说

> struct 意味着被动的对象（passive objects），它用来保存公有成员，且基本上没有成员函数。
>
> class 是动态的对象（active objects），它用来保存私有成员，你只能通过接口操作 class 对象

一般对 struct 的命名约定是，在名字后面加后缀 `_t`，而成员命名没有任何修饰，如下：

```
struct Person_t {
    string name; // public
};
```

相对来说，class 定义的类名一般不会有任何修饰，而成员名以 `_` 下划线结尾，如下

```java
class Person {
    string name_; // private
};
```

在很久以前，C++ 程序员喜欢在成员前加 `m_`，如 `m_name`，不久之后，程序员就想，与其用两个符号来修饰成员名，为什么不只用一个符号呢？而之所以不在成员名前加 `_` 前缀，是因为 C++ 编译器和内部实现是使用 `_` 或 `__` 来标识成员的，于是就有了使用 `_` 后缀来修饰成员的约定。

因为 C++ 的封装性，我们不能直接操作 class 中的私有成员，但有时候为了读写它们，我们只能定义 `setter` 和 `getter` ，如下

```java
class Person {
    string name_; // private
public:
    string name() const { return name_; } // getter 
    void set_name(const string& name) { name_ = name; } // setter 
};
```

即便如此，我们却依然要避免过多的定义 `setter` 和 `getter` 函数，因为过多的 `setter` 和 `getter` 就代表这些成员要被外部直接使用，既然这样，这些成员是否真的属于这个类呢？是不是要重新思考下此处的设计呢？例如将部分成员移到别处，作为其他类的私有成员，同时能减少 `setter` 和 `getter` 定义。



参考：

* https://www.youtube.com/watch?v=qJ4Kzk6mnFc