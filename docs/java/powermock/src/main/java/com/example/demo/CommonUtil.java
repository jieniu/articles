package com.example.demo;

import java.time.Duration;
import java.util.Date;

public class CommonUtil {
    public static long getTimeInterval(Date lastTime) {
        long duration = Duration.between(lastTime.toInstant(),
                new Date().toInstant()).getSeconds();
        return duration;
    }
}
