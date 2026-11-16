# Runtime configuration

## 可运行边界

公开源码能够独立运行CPU框架测试、任务格式验证和标记为Mock的几何图协议演示；这些命令不依赖原实验工作区。

真实双链路需要已配置的外部模型与后端。当前仓库没有“安装Python包后即可完成全部真实生成”的承诺。内部Thrift服务未公开，缺失依赖不能作为执行成功，也不能切换到Mock。

## 模型与隔离环境

`modules/generator/catalog.py`只加载元数据。默认`configs/generator_workers.json`为空，所以不会声明真实模型已就绪。维护人在`configs/generator_workers.local.json`配置可执行文件、Repo、Checkpoint、GPU与必需文件，并使用`ATTACKER_WORKER_CONFIG`指定它；这是本地管理员配置，不从任务JSON或HTTP客户端接收。

每个Worker仍通过原有固定argv＋JSON IPC调用。主环境只需Pydantic、NumPy、Pillow；Torch/Paddle/ONNX/OpenCV/dlib等按具体工具在独立模型环境准备。不要将多个旧模型环境强行合并。

公开工具元数据中的`local_eval_status=not_run`不会因为文件存在就被改写。完成本机真实Smoke后，将元数据另存为`configs/generator_catalog.local.json`，填入实际验收状态、证据和已核实耗时，并以`ATTACKER_TOOL_CATALOG`指定。`availability`由资产检查决定，`local_eval_status`由真实验收决定；二者都不能仅为通过检查而随意修改。未通过验收时，原Orchestrator仍拒绝真实内容执行。

| 工具 | 真实运行依赖 | 本项目历史环境口径 |
|---|---|---|
| InSwapper | InsightFace、InSwapper128 ONNX、检测/识别权重；检测代码依赖LivePortrait中的InsightFace模块 | CPU ONNX路径 |
| SimSwap | 官方Repo、Generator、ArcFace、Parsing、检测模型 | 隔离PyTorch GPU环境 |
| BlendFace | 官方BlendSwap代码/权重、REFace对齐、dlib68、InsightFace检测 | 隔离PyTorch GPU环境；保留原生Mask回贴 |
| LivePortrait Static | 官方Repo、F/M/G/W、stitching、landmark、InsightFace权重 | 隔离PyTorch GPU生成；CPU Source分析 |
| MobileFaceSwap（独立扩展） | 官方Repo与权重、Paddle/ONNX环境 | 本地曾使用Paddle2.4.2兼容环境；不是无差异的原Paddle2.1.2重现 |
| UniFace（独立扩展） | 独立Swapping模型、原CUDA算子、对齐/分割辅助资产 | 本地Torch1.12.1＋CUDA11.6兼容环境；原生256脸块＋显式回贴 |

默认主链路只调度InSwapper、SimSwap、BlendFace和LivePortrait。后两个新增方法仍需显式Registry绑定、资产校验和单独Smoke；没有因为保留源码就默认准入。

Worker示例结构如下，所有`/absolute/path/...`都必须替换为维护人的真实路径。此JSON是配置结构说明，不是已可执行的部署：

```json
{
  "liveportrait_static": {
    "python": "/absolute/path/to/isolated-env/bin/python",
    "repo": "/absolute/path/to/LivePortrait",
    "checkpoint": "/absolute/path/to/LivePortrait-weights",
    "gpu": "YOUR_EXPLICIT_GPU_ID",
    "timeout_s": 90,
    "required_files": ["/absolute/path/to/a/required/model.pth"]
  }
}
```

InSwapper的历史loader使用Repo之间的相对布局；BlendFace的公开路径整理只替换本机根目录，不改变对齐或生成数学。为复用这些loader，需要布局：

```text
runtime/
├── representative_repos/
│   ├── Deepfake/Face_Swapping/{insightface,SimSwap,REFace}/
│   └── Synthetic/Avatar_Digital_Human/LivePortrait/
└── caches/representative_models/{InSwapper,SimSwap,REFace,BlendFace,LivePortrait}/
```

这些目录和环境、模型均不随仓库分发。独立UniFace Worker还需要本地`support_models.json`，将对齐/分割实现与权重的路径、SHA、Mask标签、feather参数显式绑定；不能借用不存在的历史Run。第三方代码和权重应按官方说明及许可自行准备。

## 本地部署注入

[examples/runtime_factory.py](../examples/runtime_factory.py)是现有`DualBranchPipeline`的绑定示例。维护人提供被Git排除的`deployment_local.py`，至少包含：

| 名称 | 真实契约 |
|---|---|
| `input_policy` | `shared.schemas.InputPolicy`；仅允许授权输入根目录 |
| `analyzer` | `.analyze(artifact, raw_bytes)`返回带SHA绑定的`ModuleResult`；可组合HOG、LP、NISP原始观测 |
| `face_detector` | `FacePresenceDetector.detect(artifact, raw_bytes)`，给原Gate1使用 |
| `detector` | `DetectorClient.infer_one(artifact)`返回`DetectorResult`，保留连续分数、状态、阈值和版本来源 |
| `swapping_registry(tool)` | 返回真实的原`GeneratorRegistry`；BlendFace使用原`NativeLineageGenerator`绑定 |
| `gpu_guard()` | 对显式设备作当前资源准入，返回bool；不得擅自停止其他进程 |
| `batch_limits` | 本地批准的`(最大Swapping任务数, 最大Editing任务数)` |

原始接口在[shared/contracts.py](../shared/contracts.py)，输入、Artifact与结果结构在[shared/schemas.py](../shared/schemas.py)。本轮没有复制私有服务协议来伪装可移植部署。

Source筛选默认关闭。完成真实服务/资产Smoke与小规模人工核验后，维护人创建`configs/source_rules.local.json`，明确启用自己核验的受控配置。公开的阈值不是通用生产阈值：

| 观测 | 历史Pilot数值 |
|---|---|
| 单脸 | 受控输入下HOG==1 **或** LP==1；保留两者原始计数 |
| 闭嘴/张嘴 | LP lip≤0.03且mouth≤0.2 / LP lip≥0.15且mouth≥0.5 |
| 睁眼 | 两眼ratio≥0.2、差值≤0.1且blink≤0.2 |
| 正脸 | \|yaw\|≤8°、\|pitch\|≤20°、\|roll\|≤15°，frontal≥0.8且left/right≤0.2 |
| 姿态Fallback | 仅满足基本可用性及姿态条件时，frontal≥0.5；仍只选择左右转头 |

未知、冲突和后端失败保持Review/Error，不能以`1-mouth`或`1-blink`充当闭嘴/睁眼真值。左右均按图像/观察者视角；绝对yaw正值指图像左侧，负值指右侧。

冻结基础Profile保留张嘴0.2/0.35/0.5、闭嘴0、闭眼0、左右yaw±25/±35/±45。历史交付中另有经人工选择的0.6/0.7/0.8增强张嘴批次，该批次和输入不随包发布；公开基础配置没有冒充该批次参数。`keep_frontal`仅保留历史兼容文件，最终五动作规则会拒绝它；生成后Analyzer仍关闭。

## Detector与NISP服务

内部曾使用Docker Detector＋Thrift、独立CPU动作分类服务；镜像、IDL/stubs、模型和Registry都不分发。没有Docker build/pull/start、宿主机权限、SSH会话或挂载整个数据盘的公开脚本。

`configs/detector.example.json`仅展示loopback地址、端口、超时、batch≤8、实验阈值等字段。它不是正在运行的服务发现结果，`configured_service_version=unconfigured`。外部真实客户端必须保持成功/失败/未评分区分，并使用原`DetectorResult`。

NISP客户端要求显式注入维护人有权使用的stubs。`frontal/left/right/mouth/blink`是分类分数，不是校准概率；NISP不直接给face_count、yaw或眼口ratio，不能替代LP连续测量或人工状态标签。

## 输出与恢复

每批保存`results.jsonl`和`summary.json`，每个任务目录由原Pipeline保存输入快照、Source分析、执行计划、Run、Artifact、Gate1和Detector结果（如确实执行）。Generator、Modified与最终Artifact保持分别引用，图片不覆盖，分数绑定实际Artifact/SHA。

核验图可使用原`modules.visualization.sheets`，展示须遵循逐图`display_allowed`。历史Contact Sheet和业务图片不随源码公开；CLI也不会在未知展示授权下自动公开图片。

当前薄CLI拒绝覆盖或隐式恢复已有目录，不提供按分数重试、无声替换Source或强行补齐数量。未完成批次由维护人依据已保存记录另行冻结剩余任务；请勿删除原始记录后重跑同批。
