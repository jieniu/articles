# public, protected 和 private 关键字

C++ 中的继承有 3 种方式，分别是 public、protected 和 private，这三种方式分别对应不同的父类成员的访问权限，总结如下：

1. public, protected 和 private 子类都不能访问父类的 private 成员
2. public 作用域下，父类的 public 成员会被继承为 public，父类的 protected 成员会被继承为 protected
3. protected 作用域下，父类的 public 和 protected 成员会被继承为 protected 成员
4. private 作用域下，父类的 public 和 protected 成员会被继承为 private 成员

这 4 条规则实际上只有第 2 条是常用的，下面说下这 3 个继承作用域的使用场景。

## public 继承是一种 is-a 关系

我们经常会将子对象强制转换（casting）为父对象，而 public 继承在这种强制转换的场景下是无障碍的，这种情况下，子类对象可以理解为一种特殊的父类对象，即它们是一种 is-a 的关系；除此之外，其他的由 protected 和 private 作用域继承而来的对象就不具备这样的关系，下面是一个简单的例子：

```java
#include <iostream>
using namespace std;

class B {
private:
    int val_;
public:
    B(int val) : val_(val) {}
    void print_val() { cout << "val_ = " << val_ << endl; }
};

class D_pub : public B {
public:
    D_pub(int val)
    : B(val)
    {}
};

class D_pro : protected B {
public:
    D_pro(int val)
    : B(val)
    {}
};

int main() {
    D_pub pub(1);
    B* b = &pub;
    b->print_val();
  	
  	D_pro dpro(1);
  	B* b2 = &dpro; // error: 'B' is an inaccessible base of 'D_pro'
}
```

上面例子中，类 B 是一个基类，D_pub 是一个使用 public 作用域的子类，而 D_pro 是使用 protected 作用域的子类，我们在 main 中分别创建 D_pub 的对象 pub 和 D_pro 的对象 dpro，并分别赋值给父类指针，可以看到，将 D_pro 对象赋值给父类指针的语句报编译错误，原因在于类 B 中的可访问成员在 D_pro 中变成了不可访问成员，即 D_pro 对象不再是一个特殊意义的 B 对象，它们之间不具备 is-a 关系。以此类推，private 继承的子类和父类也没有 is-a 关系。

## protected 和 private 继承是一种 has-a 关系

protected 和 private 继承类似于组合模式（composition），它是一种 has-a 关系，我们看一个组合模式的例子：

```java
class hat {
public:
    void wear() {}
};

class child {
    hat h_;
public:
    void hat_wear() { h_.wear(); }
};
```

上面的代码中，child 类是以将 hat 组合进来的方式实现的，即让 child 类也具有 hat 的方法，一种很好的办法是将 hat 作为 child 的一个成员，所以这两个类具备 has-a 的关系。下面我们看使用 protected 或 private 继承如何实现 has-a 的关系：

```java
class child : private hat {
public:
    using hat::wear; // 此时就可以调用 child 对象就可以调用 hat::wear 方法了
};

int main() {
  child c;
  c.wear();
}
```

我们将 child 以 private 的方式继承自 hat，并将 `hat::wear()` 方法放置在 child 的 public 作用域中，这样 child 就「拥有了 hat 的能力」，它们之间也是一种 has-a 关系。

虽然不同的实现达到了相同的效果，但依然不建议使用 private 或 protected 的方式实现 has-a 的关系，而建议更多的使用组合模式，一是因为组合模式更为直观，其二是因为组合模式将组合的对象解耦（它们没有多一层继承关系），其三是组合模式更为灵活，试想一个类有多个组合对象的情况。

## 总结

以上，我们介绍了 C++ 中继承的三种作用域，此时我们需要记住 4 个规则：

1. public, protected 和 private 子类都不能访问父类的 private 成员
2. public 作用域下，父类的 public 成员会被继承为 public，父类的 protected 成员会被继承为 protected
3. protected 作用域下，父类的 public 和 protected 成员会被继承为 protected 成员
4. private 作用域下，父类的 public 和 protected 成员会被继承为 private 成员

以及 2 个使用场景：

1. public 继承是一种 is-a 的关系
2. private 或 protected 继承是 has-a 关系，且在实现 has-a 时，尽量使用组合模式





参考：[Advanced C++: Inheritance - Public, Protected, and Private](https://www.youtube.com/watch?v=EYuPBkgJtCQ&list=PLE28375D4AC946CC3&index=19&t=0s)