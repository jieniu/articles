package com.example.demo;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mockito;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;
import org.powermock.api.mockito.PowerMockito;
import org.powermock.api.support.membermodification.MemberModifier;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.junit4.PowerMockRunner;

import java.util.Calendar;
import java.util.Date;

@RunWith(PowerMockRunner.class)
@PrepareForTest({MyService.class, MyDao.class, CommonUtil.class})
public class MyServiceTest {
    @Test
    public void testOperate() throws IllegalAccessException {
        // 构造一个和当前调用时间永远只差 4 秒的返回值
        Calendar calendar = Calendar.getInstance();
        calendar.add(Calendar.SECOND, -4);
        Date retTime = calendar.getTime();

        // spy 是对象的部分 mock
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

    @Test
    public void testOperateWithStatic() throws IllegalAccessException {
        // 构造一个和当前调用时间永远只差 4 秒的返回值
        Calendar calendar = Calendar.getInstance();
        calendar.add(Calendar.SECOND, -4);
        Date retTime = calendar.getTime();

        // spy 是对象的部分 mock
        MyService myService = PowerMockito.spy(new MyService());
        MyDao md = PowerMockito.mock(MyDao.class);
        PowerMockito
                .when(md.getLastOperationTime(Mockito.any(long.class)))
                .thenReturn(retTime);
        // 替换 myDao 成员
        MemberModifier.field(MyService.class, "myDao").set(myService, md);

        PowerMockito.spy(CommonUtil.class);
        PowerMockito.doReturn(5L).when(CommonUtil.class);
        CommonUtil.getTimeInterval(Mockito.anyObject());
        // 假设最小操作的间隔是 5 秒，否则返回 false
        Assert.assertTrue(myService.operate(1, "test operation"));
    }
}
