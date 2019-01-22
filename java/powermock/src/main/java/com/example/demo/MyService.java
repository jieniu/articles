package com.example.demo;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.Date;

@Component
public class MyService {
    @Autowired
    private MyDao myDao;

    public MyService() {
    }

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
}
