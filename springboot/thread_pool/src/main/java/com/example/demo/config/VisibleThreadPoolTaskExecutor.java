package com.example.demo.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.util.concurrent.ListenableFuture;

import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class VisibleThreadPoolTaskExecutor extends ThreadPoolTaskExecutor {
    private static final Logger logger = LoggerFactory.getLogger(VisibleThreadPoolTaskExecutor.class);

    private void showThreadPoolInfo(String prefix) {
        ThreadPoolExecutor threadPoolExecutor = getThreadPoolExecutor();

        if (null == threadPoolExecutor) {
            return;
        }

        logger.info(prefix + ", task_count [" + threadPoolExecutor.getTaskCount()
                + "], completed_task_count [" + threadPoolExecutor.getCompletedTaskCount()
                + "], active_thread_count [" + threadPoolExecutor.getActiveCount()
                + "], blocking_queue_size [" + threadPoolExecutor.getQueue().size()
                + "], thread_pool_size [" + threadPoolExecutor.getPoolSize()
                + "], largest_pool_size_ever [" + threadPoolExecutor.getLargestPoolSize()
                + "], core_thread_pool_size [" + threadPoolExecutor.getCorePoolSize()
                + "], max_thread_pool_size [" + threadPoolExecutor.getMaximumPoolSize()
                + "], thread_keep_alive_time [" + threadPoolExecutor.getKeepAliveTime(TimeUnit.SECONDS)
                + "]");
    }

    @Override
    public void execute(Runnable task) {
        showThreadPoolInfo("1. do execute");
        super.execute(task);
    }

    @Override
    public void execute(Runnable task, long startTimeout) {
        showThreadPoolInfo("2. do execute");
        super.execute(task, startTimeout);
    }

    @Override
    public Future<?> submit(Runnable task) {
        showThreadPoolInfo("1. do submit");
        return super.submit(task);
    }

    @Override
    public <T> Future<T> submit(Callable<T> task) {
        showThreadPoolInfo("2. do submit");
        return super.submit(task);
    }

    @Override
    public ListenableFuture<?> submitListenable(Runnable task) {
        showThreadPoolInfo("1. do submitListenable");
        return super.submitListenable(task);
    }

    @Override
    public <T> ListenableFuture<T> submitListenable(Callable<T> task) {
        showThreadPoolInfo("2. do submitListenable");
        return super.submitListenable(task);

    }
}
