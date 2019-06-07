# Logic constness And Bitwise constness

### 什么是 Logic constness

有以下类 `BigArray`，其成员 `vector<int> v;` 是一个数组数据结构，为了让外部可以访问该数组，此类提供了一个 `getItem` 接口，除此之外，为了计算外部访问数组的次数，该类还设置了一个计数器 `accessCounter` ，可以看到用户每次调用 `getItem` 接口，`accessCounter` 就会自增，很明显，这里的成员 `v` 是核心成员，而 `accessCounter` 是非核心成员，我们希望接口 `getItem` 不会修改核心成员，而不考虑非核心成员是否被修改，此时 `getItem` 所具备的 `const` 特性就被称为 **logic constness**。

```c++
class BigArray {
    vector<int> v; 
    int accessCounter;
public:
    int getItem(int index) const { // 因为bitwise constness，所以无法编译通过
        accessCounter++;
        return v[index];
    }
};
```

### 什么是 Bitwise constness

但是，上面的代码不会通过编译，因为编译器不会考虑 logic constness ，于是就有了 **bitwise constness** 这个术语，可以理解为字面上的 const 属性。为了解决这种矛盾，可以把 `accessCounter` 声明为 `mutable` 的成员，即

```c++
class BigArray {
	// ...
    mutable int accessCounter; 
    // const_cast<BigArray*>(this)->accessCounter++; // 这样也行，但不建议这么做
    // ...
};
```

此时编译器是可以通过编译的。

反过来，如果你的成员是指针类型，在函数中我们修改了指针所指的数据，此时编译器依然只会维护 bitwise constness，即便我们将这样的函数声明为 const，依然是没有问题的，例如

```c++
class BigArray {
    int* v2;
	// ...
  	void setV2Item(int index, int x) const {
        v2[index] = x;
  	}
};
```

但逻辑上，这个函数不应该被声明为 const，所以这里最好把 const 关键字去掉。



**结论**

logic constness 和 bitwise constness 的重要性排序：logic constness > bitwise constness



参考：https://www.youtube.com/watch?v=8A5AwX6XExw&index=4&list=PLE28375D4AC946CC3&t=0s