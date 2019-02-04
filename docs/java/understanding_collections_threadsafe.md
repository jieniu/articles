# 理解 Java 中的 Colleciton 和线程安全

用 Java 编程比较便捷的原因之一就是它提供了丰富的类库和具备庞大的开发生态，需要实现的任何一个功能，你都可以找到合适的“工具包”，即便是这样，你也不能盲目的使用它们，不然可能会陷入性能陷阱，而今天说的 Collection 类，就是一个这样的例子。

### 1. 线程安全的陷阱

Java 中的 Collection 类中，分为线程安全的和非线程安全的，其中 `Vector` 和 `Hashtable` 属于前者，而其他的例如 `List`、`Set`、`Map` 等并不提供线程安全性，初学 Java 的同学看到这里，可能会想：这很简单嘛，在多线程环境下用 `Vector` 呗，而如果是单线程程序的话，用 `List` 就好。这种想法实际上是有问题的，虽然这样使用 Collection，在写程序的时候简单了，但如果应用对性能要求很高的话，这种选择往往是错误的。

可以看到大多数 Collection 类都没有实现线程安全，究其原因是线程安全的特性的代价非常高，而 `Vector` 和 `Hashtable` 存在的原因并不是为了解决并发性问题的，实际上它们只是较早期 Java 版本中的特性而已，所以我们在使用 Collection 时，应该避免使用这两个类。

### 2. Fail-Fast 迭代器

迭代器 iterator 一般用来对 Collection 进行遍历、修改 Collection 元素等，而当多个线程共享同一个 Collection 对象时，如果其中一个线程通过迭代器遍历 Collection，于此同时有另一个线程正在修改它，这时 Java 便会抛出 `ConcurrentModificationException` 异常，这也是 Fail-Fast 名字的出处。

Java 做这样的处理是因为多个线程同时读写迭代器是非常危险的行为，会导致程序的不确定性和一致性的问题，一旦发现此类行为，越早“制止”越好，这种机制可以很好的帮助诊断 bug，所以你的程序遇到 `ConcurrentModificationException` 这种异常应该直接终止，而不是去捕获它。

### 3. Synchronized Wrappers

如果要在多线程环境下使用 Collections，你可以利用 Synchronized Wrappers，它的形式如下：

```java
Collections.synchronizedXXX(collection);
```

`XXX`可以为 `List`、`Map`、`Set`、`SortedMap` 及 `SortedSet`，例如对于 `List` ：

```java
List<String> safeList = Collections.synchronizedList(new ArrayList<>());
```

这样，你的 Collection 就是线程安全的，即便如此，这些线程安全的 Collection 的 iterator 仍然不是线程安全的，要使 iterator 也处于线程安全状态，你需要使用 `synchronized` 代码块：

```java
synchronized (safeList) {
	Iterator<String> iterator = safeList.iterator();
    while (iterator.hasNext()) {
        String next = itorator.next();
        // ..
    }
}
```

可以看到，这种 `synchronized` block 的开销很大，因为同一时刻，只能有一个线程能运行 block 中的代码，而其他线程都只能在 block 外等待。

### 4. 并行的 Collections

Java 5 引入了并行的 Collections，它们被包含在 `java.util.concurrent` 中，且有以下 3 种并发机制：

#### 4.1 copy-on-write collections

这种 collection 存储在不可变 (immutable) 的 collection 中，任何对该 collection 的修改都会产生一个新的 collection，你可能已经猜到了，这种 collections 只适用于读远远多于写的场景。

可以预料的是，它们的 iterator 也是只读的，具备 copy-on-write 的 collections 有 `CopyOnWriteArrayList` 和 `CopyOnWriteArraySet`。

#### 4.2 Compare-And-Swap（CAS）collections

我们在很多场合都有听过 CAS 的概念，例如 Memcache 和 Mysql 中都实现了 CAS 机制：当我们要去更新一个值，先获取它的副本，然后在此副本的基础上计算出结果，最后拿结果和副本去修改原值，此时如果发现副本和原值发生了不一致，说明有其他线程抢先一步更新了原值，则更新失败，否则更新成功。

具备 CAS 机制的 Collections 包括 `ConcurrentLinkedQueue` 和 `ConcurrentSkipListMap`

#### 4.3 java.util.concurrent.lock.Lock

Lock 库提供丰富的锁机制，包括可重入锁 `ReentrantLock`，可重入的读写锁 `ReentrantReadWriteLock`，以及条件变量 `Condition`，除此之外，和 Synchronized Wrappers 的区别是，该库还提供更细粒度的 Collections 锁，即将一个 Collection 分为多个部分，每部分对应一个锁，可以显著的提高并发能力。

例如 `LinkedBlockingQueue` 提供了队首和队尾两把锁，这样你可以并行的入队和出队。其他的并发 Collections 还包括 `ConcurrentHashMap` 和 `BlockingQueue` 的所有实现。



参考：

* [Understanding Collections and Thread Safety in Java](https://www.codejava.net/java-core/collections/understanding-collections-and-thread-safety-in-java)
* [Guide to java.util.concurrent.Locks](https://www.baeldung.com/java-concurrent-locks)

