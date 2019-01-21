package com.example.demo;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mockito;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;
import org.powermock.api.mockito.PowerMockito;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.junit4.PowerMockRunner;

@RunWith(PowerMockRunner.class)
@PrepareForTest({MyTask.class})
public class MyTaskTest {
    @Test
    public void testUpdateStatus() throws InterruptedException {
        // 初始化被测对象
        MyTask myTask = PowerMockito.spy(new MyTask());
        myTask.setOperation(new MyTask.Operation());
        // 使用 doAnswer 来 mock upDateStatus 函数的行为
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
}
