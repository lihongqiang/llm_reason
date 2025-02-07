#!/bin/bash

# 以下命令作为打包示例，实际使用时请修改为自己的镜像地址, 建议每次提交前完成版本修改重新打包
# docker build -t registry.cn-shanghai.aliyuncs.com/lhq/test:0.1 . 
# docker push registry.cn-shanghai.aliyuncs.com/xxxx/test:0.1

docker login --username=415200973@qq.com registry.cn-hangzhou.aliyuncs.com
# lhqlhq123
docker build -t rregistry.cn-hangzhou.aliyuncs.com/hongqiangli/tianchi_submit:0.1 .