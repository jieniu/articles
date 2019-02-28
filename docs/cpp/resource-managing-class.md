# 指针类型成员的陷阱

当一个类拥有另一个类的指针类型的成员时，要注意隐式复制构造函数或赋值操作符产生的陷阱，如下面的程序：

```Java
class Person {
public:
    Person(string n) {
        name_ = new string(n);
    }
    ~Person() {
        delete name_;
    }
private:
    string* name_;
};

int main() {
    vector<Person> vp;
    vp.push_back(Person("Geoge"));
    vp.back().printName();

    return 0;
}
```

`Person` 类中有一个 `String` 类型的指针成员，该成员在 `Person` 的构造函数中通过 `new` 初始化，并在析构函数中 `delete`；在 `main` 中，我们定义了一个 `vector`，同时向其中插入一个 `Person` 对象，然后调用该对象的 `printName()` 函数，可以看到，这个程序非常简单，每一条语句都很直白。

可是，当运行编译后的可执行文件时，程序却崩溃了：

```
resource_manage(84052,0x1148e35c0) malloc: *** error for object 0x7fca7dc02c80: pointer being freed was not allocated
resource_manage(84052,0x1148e35c0) malloc: *** set a breakpoint in malloc_error_break to debug
[1]    84052 abort      ./resource_manage
```

崩溃的原因在这一行

```
vp.push_back(Person("Geoge"));
```

这一行代码做了 3 件事情：

1. 调用 `Person` 类的构造函数，构造一个临时的 `Person` 对象
2. 调用 `Person` 类的默认复制构造函数，将复制后的对象插入到 vector 中
3. 析构临时的 `Person` 对象

注意第 2 步的复制操作，因为默认复制构造函数属于浅层复制（shallow copy），所以第 2 步完成后，vector 中 `Person` 对象的 `name_` 指针和临时的 `Person` 对象中的 `name_` 相同，当第 3 步析构结束后，vector 中 `Person`  对象的 `name_` 指针必然就变成了野指针，直接造成了后面程序的崩溃。

如何解决这种问题呢？有两个办法：

1. 定义复制构造函数（copy constructor）和赋值操作符（copy assignment operator），对该指针成员进行深度拷贝
2. 禁止复制构造函数（copy constructor）和赋值操作符（copy assignment operator）。如果一定有复制需求，可以另外定义 `clone` 方法

我们先来看第 1 种解决方案，代码如下：

```java
class Person{
    // ...
        Person(const Person &rhs) {
            name_ = new string(*rhs.name_);
        }
        Person& operator=(const Person &rhs) {
            if (this == &rhs) {
                return *this;
            }
            string* tmp_ = name_;
            name_ = new string(*rhs.name_);
            delete tmp_;
            return *this;
        }
    // ...
};
```

通过定义复制构造函数和赋值操作符，我们重新编译并运行该程序，程序不再崩溃，根本原因是这两个自定义函数实现了成员的深度拷贝（deep copy）；即便如此，这种方法仍然不够高效，究其原因在于引入了不必要的内存拷贝和释放操作。

所以，更常规的方法是第 2 种，它从更精简的代码和更高效两方面胜出，代码如下：

```java
class Person{
    public:
        Person(string n) {
            name_ = new string(n);
        }
        ~Person() {
            delete name_;
        }
       void printName() { cout << *name_ << endl; }
    private:
        string* name_;
        Person(const Person& rhs);
        Person& operator=(const Person& rhs);
};

int main() {
    vector<Person*> vp;
    vp.push_back(new Person("Geoge")); 
    vp.back()->printName();
    delete vp.back();

    return 0;
}
```

在第 2 种方法中，我们令 `Person` 对象为不可复制的，即将复制构造函数和赋值操作符声明在 `private` 作用域中，且不定义它们；而又由于 STL 容器要求其中的对象必须具备可复制性，所以我们将这里改为往 vector 中存放指针。有时候，我们不得不复制对象，此时，我们可以为 `Person` 类中增加 `clone` 函数，让调用者显示的通过调用 `clone` 来复制对象，而不是通过隐式拷贝的方式，毕竟后者更容易出现 bug：

```java
		// ...
		Person& clone(const Person &rhs) {
            if (this == &rhs) {
                return *this;
            }
            string *tmp = name_;
            name_ = new string(*rhs.name_);
            delete tmp;
            return *this;
        }
        // ...
```

总结：本文主要介绍了当成员是指针类型时，程序很容易因为隐式拷贝而产生 bug，同时我们介绍了 2 种解决方案，其中第 2 种是更为常用的方法——禁止拷贝构造函数和赋值操作符，并在需要拷贝时，显示调用 `clone` 函数。

参考：https://www.youtube.com/watch?v=juQBXTNz4mo&list=PLE28375D4AC946CC3&index=14