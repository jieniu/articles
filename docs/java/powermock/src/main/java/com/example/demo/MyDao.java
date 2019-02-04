package com.example.demo;

import org.apache.ibatis.annotations.Mapper;
import org.springframework.stereotype.Repository;

import java.util.Date;

@Mapper
public interface MyDao {
    /**
     * 根据用户 id 查看他最近一次操作的时间
     */
    Date getLastOperationTime(long userId);

    /**
     * 获得状态
     */
    String getStatus(long operationId);
}
