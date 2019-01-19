# MySQL 配置及参数汇总

1. rows_examined

   存储在慢查询日志中，是执行器用于衡量语句扫描行数的值

2. query_cache_type

   查询缓存类型，建议设置为 DEMAND，表示按需使用查询缓存，例如

   ```mysql
   select SQL_CACHE * from T where ID=10;
   ```

3. wait_timeout

   MySQL 连接超时，默认为8小时。为了防止长连接导致的内存暴涨，建议定期重连 MySQL，或在 MySQL 5.7 及之后，调用 `mysql_reset_connection` 重新初始化链接。




