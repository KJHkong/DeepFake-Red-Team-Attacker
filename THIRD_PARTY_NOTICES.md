# Third-party implementation and model references

本目录只整理本项目的框架与集成代码，不包含下列完整官方Repo、Checkpoint、训练集或私有服务。Adapter中的预处理、原生推理、对齐与回贴方法依赖相应上游实现；保留引用不代表已获得再分发或商业使用许可。

| 项目 | 官方来源 | 与本工程的关系 |
|---|---|---|
| InSwapper / InsightFace | https://github.com/deepinsight/insightface | ONNX换脸、身份/检测；模型许可与代码许可可能不同 |
| SimSwap | https://github.com/neuralchen/SimSwap | 生成、检测与分割辅助资产 |
| BlendFace / BlendSwap | https://github.com/mapooon/BlendFace | 身份迁移、原生生成与Mask |
| LivePortrait | https://github.com/KlingAIResearch/LivePortrait | 单图数值retargeting、裁剪、stitching、回贴及Source观测 |
| REFace | https://github.com/Sanoojan/REFace | FFHQ对齐辅助实现；需要自行确认对应版本与许可 |
| MobileFaceSwap | https://github.com/Seanseattle/MobileFaceSwap | 独立复现的Paddle换脸Worker |
| UniFace | https://github.com/xc-csc101/UniFace | 独立Swapping模型；完整画布回贴为本工程显式扩展 |

REFace仅作为其他Adapter的对齐依赖，不代表它属于最终主链路三工具。未打包的Detector/NISP由维护人依据组织授权部署，不授予访问内部镜像、模型或IDL的权限。

Pydantic、NumPy、Pillow、pytest、可选Apache Thrift及各模型环境中的依赖仍受各自许可约束。本项目整体许可证尚待发布负责人确认，本整理版没有擅自添加MIT、Apache-2.0或其他授权声明。

## Demo video audio

[演示视频](docs/assets/redteam_demo.mp4)是对已公开Showcase的40秒产品展示，不是实时推理录像。

- BGM：**House Vibez — Lily J**，来自[Mixkit](https://mixkit.co/free-stock-music/house/)，使用[Mixkit Stock Music Free License](https://mixkit.co/license/modal/musicFree/)；成片截取40秒并作淡入淡出、响度与音量处理。音乐许可不等于允许单独再分发原始音轨，本仓库仅保留完成的演示视频。
- 短音效：CC0素材；具体来源与作者见[视频音效来源](docs/video_audio_credits.md)。
