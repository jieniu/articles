# 异常处理和析构函数

C++ 对待异常处理有两个规则

1. 如果在 `try...catch` 中有异常抛出，则在 `catch`  执行前，会先将 `try` 语句块对应的栈清空
2. C++ 不允许在同一个 `try...catch` 中处理1个以上的异常，如果发生此种情况，程序就会崩溃

为便于理解，我们先来看一个例子

```c++
class Dog {
public:
	string name;
	Dog(string name) {this->name = name; cout << name << " is born.\n"; }
    ~Dog() { cout << name << " is destroied.\n"; }
    
    void bark() { cout << name << " is barking.\n"; }
};

int main() {
    try {
        Dog dog1("Henry");
        Dog dog2("Bob");
        throw 20;
        dog1.bark();
        dog2.bark();
    } catch (int e) {
        cout << e << " is caught" << endl;
    }
}

/*
 * output：
 * Henry is born.
 * Bob is born.
 * Bog is destroied.
 * Henry is destroied.
 * 20 is caught
 */
```

从上面的例子可以看出，`catch` 语句块在两个局部对象析构完成后才执行，意味着在异常被捕获之前，`try` 代码块中的栈需要被清理。

我们把上面代码稍作修改，把 `throw` 语句放到析构函数中，看下会发生什么

```c++
class Dog {
    public:
        string name;
        Dog(string name) {this->name = name; cout << name << " is born.\n"; }
        ~Dog() { cout << name << " is destroied.\n"; throw 20;}

        void bark() { cout << name << " is barking.\n"; }
};

int main() {
    try {
        Dog dog1("Henry");
        Dog dog2("Bob");
        dog1.bark();
        dog2.bark();
    } catch (int e) {
        cout << e << " is caught" << endl;
    }
}

/* Output: 
 * Henry is born.
 * Bob is born.
 * Henry is barking.
 * Bob is barking.
 * Bob is destroied.
 * Henry is destroied.
 * libc++abi.dylib: terminating with uncaught exception of type int
 * [1]    51549 abort      ./exception
 */
```

可以看到程序崩溃了，崩溃原因是 `terminating with uncaught exception of type int` ：异常没有被处理。我们来分析下其中的原因，在 `try` 中，我们定义了两个对象，并按照顺序调用了它们的 `bark` 接口，随后离开 `try` 代码块，此时，编译器会自动释放这两个局部对象，调用它们的析构函数，因为栈的特性是后进先出，所以先析构 `Bob`，在执行 `Bob` 的析构函数时，抛出了异常，但此时并不会立即执行 `catch` 语句块，根据上文提到的第一条规则：「在 `catch` 执行前，需要先清理 `try` 中的堆栈」。于是 `Henry` 也被析构了，这让 `try` 语句块中抛出了 2 个异常，直接导致了程序的崩溃。

找到上述程序崩溃的元凶后，我们便学到了一条宝贵的 C++ 经验：

> 不要在析构函数中抛出异常。

因为如果你的析构函数中有异常抛出的话，你便无法控制 `try` 语句中抛出来的异常数量——这将是一场灾难。

为了不在析构函数中抛出异常，一般有两种做法：

1. 在析构函数内部捕获异常，防止异常被抛出，例如下面的代码

   ```c++
   ~Dog {
       try {
           // may throw exception
       } catch (MyException e) {
           // Catch exception
       } catch (...) {
           // 尽量不要使用 ... 来捕获异常
       }
   }
   ```

   虽然这样做，你的析构函数再不会抛出异常了，但却带来了一些隐患，即你使用了 `(...)` 来捕获异常，这种代码没有任何用处（它无法输出有效的异常信息），同时由于它会捕获一切异常，于是会将一些必要的程序缺陷掩盖起来，而不是“尽早的暴露问题”。所以在这里，我们学到的第二条经验是：

   > 不要使用 `(...)` 来捕获异常

2. **保持析构函数简洁，将可能导致异常的代码移到其他的函数中**。这也是推荐的做法。



参考

* [Advanced C++: Exceptions in Destructors](https://www.youtube.com/watch?v=LQMYwvM8RF8&t=14s)