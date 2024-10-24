# 检测头修改日志

## 检测头修改位置记录

+ 自定义检测头在如下位置添加导入 & 调用

    + 检测必要部分
        ```python
        # /workspace/ultralytics/nn/tasks.py 265行添加检测头导入，如下
        if isinstance(m, Detect, Detect2200):

        # /workspace/ultralytics/nn/tasks.py 331行添加检测头导入，如下
        if isinstance(m, Detect, Detect2200):

        # /workspace/ultralytics/nn/tasks.py 1070行添加检测头导入，如下
        elif m in {Detect, Detect2200, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}:
        ```

        ```python
        # /workspace/ultralytics/utils/plotting.py 1314行添加检测头导入，如下
        for m in {"Detect", "Detect2200", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder"}:  # all model heads
        ```

    + 非检测必要，其他可能需要
        ```python
        # /workspace/ultralytics/engine/exporter.py 72行添加检测头导入，如下
        from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder, Detect2200

        # /workspace/ultralytics/engine/exporter.py 254行添加检测头模块调用， 如下
        if isinstance(m, (Detect, Detect2200,RTDETRDecoder)): 

        # /workspace/ultralytics/engine/exporter.py 496行添加检测头模块调用， 如下
        if isinstance(self.model.model[-1], Detect, Detect2200):

        # /workspace/ultralytics/nn/modules/__init__.py 76行添加检测头导入，如下
        from .ours_head import Detect2200
        ```

# /workspace/ultralytics/nn/tasks.py 第16行
detece改为detect2200
from ultralytics.nn.modules import (
  AIFI, C1, C2, C2PSA, C3, C3TR, ELAN1, OBB, PSA, SPP, SPPELAN, SPPF, AConv, ADown, Bottleneck, BottleneckCSP, C2f, C2fAttn, C2fCIB, C2fPSA, C3Ghost, C3k2, C3x, CBFuse, CBLinear, Classify, Concat, Conv, Conv2, ConvTranspose, Detect2200, DWConv, DWConvTranspose2d, Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, ImagePoolingAttn, Pose, RepC3, RepConv, RepNCSPELAN4, RepVGGDW, ResNetLayer, RTDETRDecoder, SCDown, Segment, WorldDetect, v10Detect
)

/workspace/ultralytics/nn/modules/ours_head/__init__.py
from .head2200 import Detect2200

/workspace/ultralytics/nn/modules/ours_head/head2200.py
class Detect2200(nn.Module) init和forward重写

/workspace/ultralytics/nn/modules/ours_head/head2200.py
311行  for a, b, s in zip(m.cv2, m.cv3_1,m.cv3_2,m.cv3_3, m.stride):






/workspace/ultralytics/nn/tasks.py
13行  
from ultralytics.nn.modules.ours_head import (
    Detect2200
)

/workspace/ultralytics/nn/modules/ours_head/head2100.py
class Segment(Detect2100):
class OBB(Detect2100):