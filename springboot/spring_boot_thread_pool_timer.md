# Spring Boot 中的线程池和 Timer 定时器

Spring Boot 是一个只写几个配置，就可以完成很多功能的 Java 框架，例如你想要一个线程池，只需两步：

1. 在应用入口 Application 主类上加注解 `@EnableScheduling`

   ```java
   @SpringBootApplication
   @EnableScheduling
   public class DemoApplication {
   	public static void main(String[] args) {
   		SpringApplication.run(DemoApplication.class, args);
   	}
   }
   ```

2. 添加一个线程池配置类，增加 `@EnableAsync` 注解

   ```java
   @Configuration
   @EnableAsync
   public class AsyncConfig {
       @Value("${async.core_pool_size}")
       private int corePoolSize;
       @Value("${async.max_pool_size}")
       private int maxPoolSize;
       @Value("${async.queue_capacity}")
       private int queueCapacity;
   
       @Bean
       public Executor taskExecutor() {
           ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
           executor.setCorePoolSize(corePoolSize);
           executor.setMaxPoolSize(maxPoolSize);
           executor.setQueueCapacity(queueCapacity);
           executor.initialize();
           return executor;
       }
   }
   ```

   其中 `corePoolSize` 是线程池中的线程数，`queueCapacity` 是线程池中队列的大小，如果队列满了，新增加的任务会被丢弃掉，为了避免队列满载，我们可以设置 `maxPoolSize`，这样会多出 `maxPoolSize-corePoolSize` 个线程来处理过载的任务。

   一般的，我们可以把 `corePoolSize` 设置为机器的核心数，而 `maxPoolSize` 为 2 倍的核心数；同时，队列满载的问题是非常严重的，说明我们程序的性能出现了问题，此时需要对程序进行优化或者扩容，于是我们需要监控这种情况的发生。

3. 监控线程池

   你可以通过继成 `ThreadPoolTaskExecutor` 类来实现该功能

   ```java
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
   ```

   上面 `showThreadPoolInfo()` 函数会打印线程池的运行时数据

   * `getTaskCount()` ：已完成的任务数
   * `getActiveCount()`：正在运行的任务数
   * `getQueue().size()`： 队列的长度
   * `getPoolSize()`：当前线程数
   * `getLargestPoolSize()`：曾经出现的最大的线程数

   此外，你还要替换这一行

   ```java
   ThreadPoolTaskExecutor executor = new VisibleThreadPoolTaskExecutor();
   ```

4. 将线程池应用在定时器上

   我们现在可以创建一个定时器，并让刚才创建的线程池来驱动定时任务，注意这里的 `@Async` 注解

   ```java
   @Component
   @Async
   public class Timer {
       private static final Logger logger = LoggerFactory.getLogger(ScheduledService.class);
       @Scheduled(cron = "0/5 * * * * *")
       public void scheduled(){
           logger.info("=====>>>>> using cron {}", System.currentTimeMillis());
       }
       @Scheduled(fixedRate = 5000)
       public void scheduled1() {
           logger.info("=====>>>>> using fixedRate{}", System.currentTimeMillis());
       }
       @Scheduled(fixedDelay = 5000)
       public void scheduled2() {
           logger.info("=====>>>>> using fixedDelay{}",System.currentTimeMillis());
       }
   }
   ```

5. demo 下载

   [在这里下载 demo 源码](https://github.com/jieniu/articles/tree/master/springboot/thread_pool)，运行试一下吧


参考：

* [Spring Boot线程池的使用心得](https://blog.csdn.net/m0_37701381/article/details/81072774)
* [ThreadPoolTaskExecutor使用详解](https://blog.csdn.net/foreverling/article/details/78073105)
* [SpringBoot几种定时任务的实现方式](http://www.wanqhblog.top/2018/02/01/SpringBootTaskSchedule/)