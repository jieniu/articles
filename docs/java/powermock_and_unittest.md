# Java 中的 UnitTest 和 PowerMock

学习一门计算机语言，我觉得除了学习它的语法外，最重要的就是要学习怎么在这个语言环境下进行单元测试，因为单元测试能帮你提早发现错误；同时给你的程序加一道防护网，防止你的修改破坏了原有的功能；单元测试还能指引你写出更好的代码，毕竟不能被测试的代码一定不是好代码；除此之外，它还能增加你的自信，能勇敢的说出「我的程序没有bug」。

每个语言都有其常用的单元测试框架，本文主要介绍在 Java 中，我们如何使用 PowerMock，来解决我们在写单元测试时遇到的问题，从 Mock 这个词可以看出，这类问题主要是解依赖问题。

在写单元测试时，为了让测试工作更简单、减少外部的不确定性，我们一般都会把被测类和其他依赖类进行隔离，不然你的类依赖得越多，你需要做的准备工作就越复杂，尤其是当它依赖网络或外部数据库时，会给测试带来极大的不确定性，而**我们的单测一定要满足快速、可重复执行的要求**，所以隔离或解依赖是必不可少的步骤。

而 Java 中的 PowerMock 库是一个非常强大的解依赖库，下面谈到的 3 个特性，可以帮你解决绝大多数问题：

1. 通过 PowerMock 注入依赖对象
2. 利用 PowerMock 来 mock static 函数
3. 输出参数（output parameter）怎么 mock

## 通过 PowerMock 注入依赖对象

假设你有两个类，`MyService` 和 `MyDao`，`MyService` 依赖于 `MyDao`，且它们的定义如下

```java
// MyDao.java
@Mapper
public interface MyDao {
    /**
     * 根据用户 id 查看他最近一次操作的时间
     */
    Date getLastOperationTime(long userId);
}

// MyService.java
@Service
public class MyService {
	@Autowired
	private MyDao myDao;
	
    public boolean operate(long userId, String operation) {
        Date lastTime = myDao.getLastOperationTime(userId);
        // ...
    }
}
```

这个服务提供一个 `operate` 接口，用户在调用该接口时，会被限制一个操作频次，所以系统会记录每个用户上次操作的时间，通过 `MyDao.getLastOperationTime(long userId)` 接口获取，现在我们要对 `MyService` 类的 `operate` 做单元测试，该怎么做？

你可能会想到使用 SpringBoot，它能自动帮我们初始化 `myDao` 对象，但这样做却存在一些问题：

1. SpringBoot 的启动速度很慢，这会延长单元测试的时间
2. 因为时间是一个不断变化的量，也许这一次你构造的时间满足测试条件，但下一次运行测试时，可能就不满足了。

由于以上原因，我们一般在做单元测试时，不启动 SpringBoot 上下文，而是采用 PowerMock 帮我们注入依赖，对于上面的 case，我们的测试用例可以这样写：

```java
// MyServiceTest.java
@RunWith(PowerMockRunner.class)
@PrepareForTest({MyService.class, MyDao.class})
public class MyServiceTest {
    @Test
    public void testOperate() throws IllegalAccessException {
        // 构造一个和当前调用时间永远只差 4 秒的返回值
    	Calendar calendar = Calendar.getInstance();
        calendar.add(Calendar.SECOND, -4);
        Date retTime = calendar.getTime();
        
        // spy 是对象的“部分 mock”
        MyService myService = PowerMockito.spy(new MyService());
        MyDao md = PowerMockito.mock(MyDao.class);
        PowerMockito
                .when(md.getLastOperationTime(Mockito.any(long.class)))
                .thenReturn(retTime);
        // 替换 myDao 成员
        MemberModifier.field(MyService.class, "myDao").set(myService, md);
        // 假设最小操作的间隔是 5 秒，否则返回 false
        Assert.assertFalse(myService.operate(1, "test operation"));
    }
}
```

从上面代码中，我们首先构造了一个返回时间 `retTime`，模拟操作间隔的时间为 4 秒，保证了每次运行测试时该条件不会变化；然后我们用 `spy` 构造一个待测试的 `MyService` 对象，`spy` 和 `mock` 的区别是，`spy` 只会部分模拟对象，即这里只修改掉 `myService.myDao` 成员，其他的保持不变。

然后我们定义了被 mock 的对象 `MyDao md` 的调用行为，当 `md.getLastOperationTime` 函数被调用时，返回我们构造的时间 `retTime`，此时测试环境就设置完毕了，这样做之后，你就可以很容易的测试 `operate` 函数了。

## 利用 PowerMock 来 mock static 函数

上文所说的使用 PowerMock 进行依赖注入，可以覆盖测试中绝大多数的解依赖场景，而另一种常见的依赖是 static 函数，例如我们自己写的一些 `CommonUtil` 工具类中的函数。

还是使用上面的例子，假设我们要计算当前时间和用户上一次操作时间之间的间隔，并使用 `public static  long getTimeInterval(Date lastTime)` 实现该功能，如下：

```js
// CommonUtil.java
class CommonUtil {
    public static long getTimeInterval(Date lastTime) {
        long duration = Duration.between(lastTime.toInstant(),
                new Date().toInstant()).getSeconds();
        return duration; 
    }
}
```

我们的 `operator` 函数修改如下

```java
// MyService.java
// ...
    public boolean operate(long userId, String operation) {
        Date lastTime = myDao.getLastOperationTime(userId);
        long duration = CommonUtil.getTimeInterval(lastTime);
        if (duration >= 5) {
            System.out.println("user: " + userId + " " + operation);
            return true;
        } else {
            return false;
        }
    }
// ...
```

这里先从 `myDao` 获取上次操作的时间，再调用 `CommonUtil.getTimeInterval` 计算操作间隔，如果小于 5 秒，就返回 `false`，否则执行操作，并返回 `true`。那么我的问题是，如何解掉这里 static 函数的依赖呢？我们直接看测试代码吧

```js
// MyServiceTest.java
@PrepareForTest({MyService.class, MyDao.class, CommonUtil.class})
public class MyServiceTest {
// ...
    @Test
    public void testOperateWithStatic() throws IllegalAccessException {
        // ...
        PowerMockito.spy(CommonUtil.class);
        PowerMockito.doReturn(5L).when(CommonUtil.class);
        CommonUtil.getTimeInterval(Mockito.anyObject());
        // ...
    }
}
```

首先在注解 `@PrepareForTest` 中增加 `CommonUtil.class`，依然使用 `spy` 对类 `CommonUtil` 进行 mock，如果不这么做，这个类中所有静态函数的行为都会发生变化，这会给你的测试带来麻烦。`spy` 下面的两行代码你应该放在一起解读，意为当调用 `CommonUtil.getTimeInterval` 时，返回 5；这种写法比较奇怪，但却是 PowerMock 要求的。至此，你已经掌握了 mock static 函数的技巧。

## 输出参数（output parameter）怎么 mock

有些函数会通过修改参数所引用的对象作为输出，例如下面的这个场景，假设我们的 operation 是一个长时间执行的任务，我们需要不断轮训该任务的状态，更新到内存，并对外提供查询接口，如下代码：

```java
// MyTask.java
// ...
    public boolean run() throws InterruptedException {
        while (true) {
            updateStatus(operation);

            if (operation.getStatus().equals("success")) {
                return true;
            } else {
                Thread.sleep(1000);
            }
        }
    }

    public void updateStatus(Operation operation) {
        String status = myDao.getStatus(operation.getOperationId());
        operation.setStatus(status);
    }
// ...
```

上面的代码中，`run()` 是一个轮询任务，它会不断更新 operation 的状态，并在状态达到 `"success"` 时停止，可以看到，`updateStatus` 就是我们所说的函数，虽然它没有返回值，但它会修改参数所引用的对象，所以这种参数也被称作输出参数。

现在我们要测试 `run()` 函数的行为，看它是否会在 `"success"` 状态下退出，那么我们就需要 mock `updateStatus` 函数，该怎么做？下面是它的测试代码：

```js
    @Test
    public void testUpdateStatus() throws InterruptedException {
        // 初始化被测对象
        MyTask myTask = PowerMockito.spy(new MyTask());
        myTask.setOperation(new MyTask.Operation());
        // 使用 doAnswer 来 mock updateStatus 函数的行为
        PowerMockito.doAnswer(new Answer<Object>() {
            @Override
            public Object answer(InvocationOnMock invocation) throws Throwable {
                Object[] args = invocation.getArguments();
                MyTask.Operation operation = (MyTask.Operation)args[0];
                operation.setStatus("success");
                return null;
            }
        }).when(myTask).updateStatus(Mockito.any(MyTask.Operation.class));

        Assert.assertEquals(true, myTask.run());
    }
```

上面的代码中，我们使用 `doAnswer` 来 mock `updateStatus` 的行为，相当于使用 `answer` 函数来替换原来的 `updateStatus` 函数，在这里，我们将 `operation` 的状态设置为了 `"success"`，以期待 `myTask.run()` 函数返回 `true`。于是，我们又学会了如何 mock 具有输出参数的函数了。



以上代码只为了说明应用场景，并非生产环境级别的代码，且均通过测试，为方便后续学习，你可以在这里下载：https://github.com/jieniu/articles/tree/master/java/powermock

参考：

* [Using PowerMock to mock/stub static void method calls in JUnit.](https://tarunsapra.wordpress.com/2011/07/31/mocking-static-void-calls-with-powermock-junit/)
* [doanswer-for-static-methods-powermock](https://stackoverflow.com/questions/18069396/doanswer-for-static-methods-powermock)
