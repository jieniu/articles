# 理解 Java 中的 Colleciton 和线程安全

用 Java 编程比较便捷的原因之一就是它提供了丰富的类库和具备庞大的开发生态，需要实现的任何一个功能，你都可以找到合适的“工具包”，即便是这样，你也不能盲目的使用它们，不然可能会陷入性能陷阱，而今天说的 Collection 类，就是一个这样的例子。

### 1. 线程安全的陷阱

Java 中的 Collection 类中，分为线程安全的和非线程安全的，其中 `Vector` 和 `Hashtable` 属于前者，而其他的例如 `List`、`Set`、`Map` 等并不提供线程安全性，初学 Java 的同学看到这里，可能会想：这很简单嘛，在多线程环境下用 `Vector` 呗，而如果是单线程程序的话，用 `List` 就好。这种想法实际上是有问题的，虽然这样使用 Collection，在写程序的时候简单了，但如果应用对性能要求很高的话，这种选择往往是错误的。

可以看到大多数 Collection 类都没有实现线程安全，究其原因是线程安全的特性的代价非常高，而 `Vector` 和 `Hashtable` 存在的原因并不是为了解决并发性问题的，实际上它们只是较早期 Java 版本中的特性而已，所以我们在使用 Collection 时，应该避免使用这两个类。

### 2. Fail-Fast 迭代器

迭代器 iterator 一般用来对 Collection 进行遍历、修改 Collection 元素等，而当多个线程共享同一个 Collection 对象时，如果其中一个线程通过迭代器遍历 Collection，于此同时有另一个线程正在修改它，这时 Java 便会抛出 `ConcurrentModificationException` 异常，这也是 Fail-Fast 名字的出处。

Java 做这样的处理是因为多个线程同时使用读写迭代器是非常危险的行为，会导致程序的不确定性和一致性的问题，一旦发现此类行为，越早“制止”越好，这种机制可以很好的帮助诊断 bug，所以你的程序遇到 `ConcurrentModificationException` 这种异常应该直接终止，而不是去捕获它。



参考：

* [Understanding Collections and Thread Safety in Java](https://www.codejava.net/java-core/collections/understanding-collections-and-thread-safety-in-java)