package com.example.demo;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Async;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
@Async
public class Timer {
    private static final Logger logger = LoggerFactory.getLogger(Timer.class);

    @Scheduled(cron = "0/5 * * * * *")
    public void scheduled(){
        logger.info("=====>>>>> using cron  {}", System.currentTimeMillis());
    }
    @Scheduled(fixedRate = 5000)
    public void scheduled1() {
        logger.info("=====>>>>> using fixedRate {}", System.currentTimeMillis());
    }
    @Scheduled(fixedDelay = 5000)
    public void scheduled2() {
        logger.info("=====>>>>> using fixedDelay {}",System.currentTimeMillis());
    }
}
