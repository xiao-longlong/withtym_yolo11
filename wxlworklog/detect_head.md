# 检测头修改日志

## 检测头修改位置记录

+ 添加自定义检测头位置

    + 检测必要部分
        ```python
        
        ```

    + 非检测必要，其他可能需要

        ```python
        # /workspace/ultralytics/engine/exporter.py 72行添加Detect头，如下
        from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder, Detect2200

        # /workspace/ultralytics/nn/modules/__init__.py 76行添加检测头，如下
        from .ours_head import Detect2200
        ```