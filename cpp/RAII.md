# RAII 技术（Resource Aquisition is Initialization）

RAII（Resource Aquisition is Initialization）技术是用对象来管理资源的一种技术，资源可以指内存、socket、IPC 等。

### 用 RAII 管理锁资源

这个概念比较抽象，我们还是从具体的例子中学习，一般我们这样使用互斥锁：

```c++
pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
void functionA() {
    pthread_mutex_lock(&mu);
    // ... 操作共享资源
    pthread_mutex_unlock(&mu); 
}
```

即，我们在使用共享资源之前通过 `pthread_mutex_lock` 加锁，并在使用完资源后，通过 `pthread_mutex_unlock` 解锁，但这种代码隐患极大，因为你不能保证锁一定会释放，例如在使用资源的时候可能抛出异常，那么这个锁就永远得不到释放，那有什么办法可以让锁一定释放，甚至自动释放呢？那就要用到今天提到的 RAII 技术：我们用对象来管理锁，对象存储在栈中，利用代码块在退出时会自动释放栈资源的特性，锁也会自动得到释放，如下面的代码：

```c++
#include <pthread.h>

pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
class Lock {
private:
    pthread_mutex_t* m_pm;
public:
    explicit Lock(pthread_mutex_t* pm) { pthread_mutex_lock(pm); m_pm = pm; }
    ~Lock() { pthread_mutex_unlock(m_pm); }
};

void functionA() {
    Lock mylock(&mu);
    // ... 操作共享资源
    // mutex会在函数退出时自动释放
}
```

上面代码中，`Lock` 构造函数接受一个 mutex 指针，同时会调用 `pthread_mutex_lock` 加锁，并会在该对象被析构时，调用 `pthread_mutex_unlock` 解锁，这就做到了对象创建时加锁，释放时解锁的效果，如果我们把这个对象放到栈中，则锁资源也会随着该对象在栈中的生命周期进行自动的加锁和解锁，函数或者代码块都可以构造这样的上下文。而这种用对象来管理资源的方式，就是我们开篇所说的 RAII。

### shared_ptr 也是一种 RAII

另一个典型的使用 RAII 技术的例子是 `std::shared_ptr`，我们通过 `shared_ptr` 来管理资源——一般是堆中申请的对象，`shared_ptr` 通过引用计数来管理指针对象，我们对 `shared_ptr` 进行复制，引用计数就加 1，相反，如果减少一个 `shared_ptr`，引用计数就减 1，当引用计数减到 0 时，会自动调用 `delete` 释放指针对象，下面的代码使用了一个 `pd` 智能指针来管理 `dog` 对象，当 `pd` 退出作用域，如果没有额外的智能指针引用 `dog`，则 `dog` 会被自动释放：

```c++
int function_A() {
	// pd 退出作用域时，dog 会自动释放
	std::shared_ptr<dog> pd(new dog());
}
```

下面我们来看一下使用 `shared_ptr` 的一个陷阱，代码如下：

```
class dog;
class Trick;
void train(std::shared_ptr<dog> pd, Trick dogtrick);
Trick getTrick();

int main() {
	train(std::shared_ptr<dog> pd(new dog()), getTrick());
}
```

函数 `train` 是一个训练函数，它接受两个参数：`dog` 和 `Trick`，即具体训练 `dog` 的方法由 `Trick` 提供，但实际上这行代码是有问题的，问题在于，编译器调用 `new dog()`、`getTrick()`  和 `shared_ptr<dog> pd()` 这三个函数的顺序是不确定的，如果编译器正好按照以下顺序来执行：

1. `new dog()`
2. `getTrick()`
3. `shared_ptr<dog> pd()`

同时在执行到第 2 步  `getTrick()` 时抛出了异常，那么 `dog` 指针就没有被智能指针管理起来，于是就发生了内存泄漏。这个问题怎么解决，我们把 `train` 这行代码拆成两行就可以了，如下：

```
int main() {
	std::shared_ptr<dog> pd(new dog());
    train(pd, getTrick());
}
```

所以

>  在初始化 `shared_ptr` 时，不要和其他语句放在一起使用

### RAII 对象的复制问题

最后，我们再来看一个 RAII 对象复制的问题，仍然是上文定义的锁 `Lock`，如果对 `Lock` 对象调用赋值构造函数，即：

```c++
Lock L1(&mu);
Lock L2(L1);
```

此时 `m_pm` 会被多个 RAII 对象持有，且因为每个 RAII 对象析构时都会对 `m_pm` 进行解锁，所以程序就无法控制该锁的解锁时机了，因此，为了解决这问题，我们首先想到的方案就是禁止 `Lock` 对象的复制能力，具体做法可以参考之前的文章[《没有学不会的 C++：禁止成员函数（disallow functions）》](https://www.jianshu.com/p/1efc919875ec)。

今天我们来学习另外一种解决方案，即使用智能指针 `shared_ptr` 来解决 RAII 锁的复制问题，思路是这样的，因为智能指针只有在引用计数减为 0 时，才执行真正的「清理」工作，如果把「清理」换成解锁，我们就不用担心多次解锁的问题。

正好，`shared_ptr` 支持用户自定义「清理」方法，如下是 `shared_ptr` 的声明

```
template<class Other, class D> shared_ptr(Other* ptr, D deleter);
```

第二个参数是引用计数为 0 时调用的「清理」函数，默认会使用 `delete`，所以在锁场景，我们把它替换为 `pthread_mutex_unlock` 即可，完整的代码如下：

```
pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
class Lock {
    private:
        std::shared_ptr<pthread_mutex_t> pMutex;
    public:
        explicit Lock(pthread_mutex_t *pm)
            : pMutex(pm, pthread_mutex_unlock) {
                pthread_mutex_lock(pm);
            }
};
```

可以看到，`Lock` 中 `pMutex` 是一个 `pthread_mutex_t` 类型的智能指针，它在构造函数被调用时初始化，且 `deleter` 是 `pthread_mutex_unlock`，同时会调用 `pthread_mutex_lock` 进行加锁，这种机制不限制 `Lock` 的复制，且只有在所有「复制品」都释放时，才自动调用 `pthread_mutex_unlock` 进行解锁，这是非常理想的使用 RAII 控制锁的方法。



参考：

* https://www.youtube.com/watch?v=ojOUIg13g3I&index=11&list=PLE28375D4AC946CC3&t=0s



