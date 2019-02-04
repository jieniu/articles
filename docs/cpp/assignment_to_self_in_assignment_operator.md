# 复制操作符怎么写

C++ 中的操作符重载可以让我们的代码更符合人们的阅读习惯，而 `operator=` 赋值操作符又是最常被重载的操作符。本篇主要谈到我们在写 `operator=` 时可能会遇到的复制相同对象的问题，及我们该如何解决它。

对于「复制自己」，你肯定不会写这样的代码

```
Dog a;
a = a;
```

但你可能会写这样的代码

```
dogs[i] = dogs[j];
```

而当 `i` 和 `j` 相等时，你便无意间写出了「复制自己」的代码，但到底「复制自己」会出现什么问题呢？先来看下下面这段代码：

```
class Collar{};
class Dog {
    Collar* pCollar;
    
    Dog& operator=(const Dog& rhs) {
        delete pCollar;
        pCollar = new Collar(*rhs.pCollar);
        return *this;
    }
};
```

上面这段赋值操作符的实现，首先就把指针 `pCollar` 给释放了，如果传入的对象就是自己，那么 `pCollar = new Collar(*rhs.pCollar)` 这一行中的 `rhs.pCollar` 就会引用一个被释放的指针，这会对程序造成灾难性的结果。

于是，为避免「复制自己」的问题，我们可以在 `delete` 之前加一个条件判断，如下

```c++
// ...
    Dog& operator=(const Dog& rhs) {
        if (this == &rhs) {
            return *this;
        }
        
        delete pCollar;
        pCollar = new Collar(*rhs.pCollar);
        return *this;
    }
// ... 
```

但这还没完，这里还有一个漏洞，如果 `new Collar(*rhs.pCollar)` 抛出异常，则该对象的 `pCollar` 由于被释放了，而变成了一个”野指针“，这也会给程序带来无法预期的结果。所以更为安全的写法是这样的：

```
// ...
    Dog& operator=(const Dog& rhs) {
        if (this == &rhs) {
            return *this;
        }
        
        Collar* pOriginCollar = pCollar;
        pCollar = new Collar(*rhs.pCollar);
        delete pOriginCollar;
        return *this;
    }
// ...
```

可以看到，即便 `new Collar` 抛出了异常，`pCollar` 所指向的内容仍然不变。

此外，我们还有另外一种解决「复制自己」的方案，即面向对象中常用的委派（delegate），例如这里要复制的 `Collar` 对象，不是在宿主 `Dog` 对象中完成，而是把复制的动作委派给 `Collar`  的复制构造函数去完成，这样做的好处是各个类的代码各司其职，复杂度降低，更不易出错，如下：

```
class Collar {
    int price;
public:
	Collar &operator=(const Collar& rhs) {
    	if (this == &rhs) {
        	return *this;
    	}
    	price = rhs.price;
    	return *this;
	}
};

class Dog {
    Collar* pCollar;
    Dog& operator=(const Dog& rhs) {
        *pCollar = *rhs.pCollar; // member by member copy of Collars
        return *this;
    }
};
```

以上，请记住一点

> 在写复制操作符时，要避免复制自己。



参考：

https://www.youtube.com/watch?v=4qhz7E59QBs&index=9&list=PLE28375D4AC946CC3