package com.example.demo;

import org.springframework.beans.factory.annotation.Autowired;

public class MyTask {
    @Autowired
    MyDao myDao;
    Operation operation;

    public void setOperation(Operation operation) {
        this.operation = operation;
    }

    public static class Operation {
        long operationId;
        String status;

        public String getStatus() {
            return status;
        }

        public void setStatus(String status) {
            this.status = status;
        }

        public long getOperationId() {
            return operationId;
        }

        public void setOperationId(long operationId) {
            this.operationId = operationId;
        }
    }

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
}
