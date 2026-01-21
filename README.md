# FunFlow

高自由度

数据处理->模型训练->推理评估->模型压缩->多格式导出

提供配置文件模板

提供使用模版

## 元数据构造 jsonl

每一行都是一条jsonline

参考

```json
{"record_id": "eval_003091", "patient_id": "ID02_pd_2_0_0_readtext", "audio_path": "/data/data/parkinson_dataset/speech_dataset/silero_sliced/Parkinson/ID02_pd_2_0_0_readtext/ID02_pd_2_0_0_readtext_part10.wav", "modality": "audio", "audio_type": null, "sampling_rate": 16000, "duration": 5.896, "transcription": null, "label": "Parkinson", "vad_segments": null, "channels": 1}
{"record_id": "eval_001754", "patient_id": "adrso274", "audio_path": "/data/data/parkinson_dataset/speech_dataset/silero_sliced/Healthy/adrso274/adrso274_part16.wav", "modality": "audio", "audio_type": null, "sampling_rate": 16000, "duration": 5.128, "transcription": null, "label": "Healthy", "vad_segments": null, "channels": 1}
```

## 在线数据管道

针对不同数据，任务，按照规范写processor即可。

processor规范：由串联的一系列生成器构成

1. parse_raw（用于从jsonl中提取你需要的信息）（必须实现）
2. 2-n:一系列你需要的生成器：例如数据读取，数据预处理，滑窗，特征提取，数据增强等等（可选实现）
3. n+1. shuffle（样本级打乱）（内部已实现）
4. n+2. batch（将样本按照指定batch_size分批）（内部已实现）
5. n+3. collate_fn（用于把数据整理成送入模型的形状并转为tensor，然后以字典形式返回，一般包含feats，labels等等键，可根据任务返回所需的任何键，比如构造对比学习样本对之类的都可以）（必须实现）

## 构建模型
模板方法模式

基类模型默认架构：preprocessing(upsampling, cmvn等等)->backbone->neck->head->loss

通过配置文件来构造模型，没有配置的模块会被定义为identity（所以最终模型可能会包含几个identity模块，对性能的影响可以忽略，后续导出的时候，比如torchscript和onnx都会自动去掉多余的identity模块）

当然你也可以选择不继承基类实现模型，或者直接把整个模型定义为backbone忽略其他所有模块即可。

核心方法：

### _parse_inputs

这是一个抽象方法，子类模型需要重写。

这个方法有两个职责：1. 接受模型输入并进行类型和形状检查；2. 对接并解析数据管道的输出，从中取出需要进入模型的数据。

### forward

这是模型的前向传播，如果使用默认模型架构可无需重写。注意输出必须是字典，你可以输出任何你需要的中间特征以及推理结果。

### forward_train

针对模型训练时候的前向，相比forward方法多了损失计算这一个环节。输出结果同样为字典，比forward方法的输出多了loss这个键，loss键对应的值可以是一个字典包含多项损失（其中必须包含 loss 字段用于表示总损失，其余字段随意，所有字段都会出现在训练监控中）也可以是一个标量tensor表示总损失。设计损失函数的时候必需遵循这个规范。

## train 和整个 trainer

train和trainer都已经构造好了，无需任何改动。

train负责编排，dataloader构造，model构造然后调用trainer进行训练

trainer的核心：

### optimizer

通过配置构造

### scheduler

通过配置构造

### hooks

这个设计是训练自由度的关键，核心思想是实现针对当前任务的代码和核心训练不变代码解耦。具体实现是：你可以在，训练开始前/结束后，每个epoch开始前/结束后，每个batch/step开始前/结束后，插入任意数量钩子实现任何你想做的事，这完全不会影响主trainer脚本。

默认钩子：

- LoggerHook：用于将训练日志写入文件，便于后续检查。
- EvalHook：用于验证集的指标评估。你需要实现针对当前任务的evaluator实例来对验证集 计算任何你感兴趣的指标，更好的监控训练。evaluator的实现需要继承基类，稍后会介绍。
- CheckpointHook：用于训练过程中的模型保存，可以设定保存数量上限，通过指定你最关注的指标来进行对比从而不断保存当前最优的指定数量模型。

可选钩子：

- EarlystoppingHook：用于避免训练过程中的过拟合，通过设定容忍度，指定你最关注的指标来进行对比，如果模型在容忍度内仍然没有改进，则训练会提前停止。
- LRSchedulerHook：用于训练预热，可以为不包含预热的学习率策略提供预热能力。
- TensorBoardHook：用于可视化监控所有训练指标，验证指标，学习率调度。
- QATHook：用于在训练过程中控制量化观察器和Norm统计的冻结时机。

### evaluator
模板方法模式

从EvalHook解耦出来根据当前任何实现，配合EvalHook计算你需要的指标

通常需要重写的方法：

- _process_batch_labels：用于从每个batch中提取出你的标签信息，batch是一个字典，标签对应的键就是你的数据管道输出规定的键，可以有多个。
- compute_metrics：（抽象方法，必须实现）用于计算所有你需要的指标。

## inference
模板方法模式

实现推理逻辑

标准化定义推理结果数据类，推理统计信息数据类（计时）

推理器基类base：后续推理需要根据当前任务继承基类实现推理器子类

核心方法：

- load_model：（抽象方法，必须实现），主要是考虑到需要加载不同格式的模型，utils中的model_loader有提供load_pytorch_model方法用于加载pytorch格式模型，提供load_onnx_model方法用于加载onnx格式模型。提供load_qat_model用于加载QAT模型。
- preprocess：（抽象方法，必须实现）处理单个样本（如果是批量的话暂时也是循环调用preprocess，待优化），输出特征用于模型推理。推荐参考配置文件中的验证数据处理方法。
- forward：（抽象方法，必须实现）模型前向推理，主要需要区分不同模型格式的推理方式。
- postprocess：（抽象方法，必须实现）逐个后处理每个样本（待优化）。
- compute_metrics：用于计算推理指标

inference_cli：命令行工具

## export
模板方法模式

实现导出逻辑

实现导出器基类base_exporter：

核心方法：

- load_model：（通常不需要重写）通过配置文件和checkpoint加载模型用于导出。
- export：（抽象方法，必须重写），实现具体的导出逻辑
- verify：可选实现，验证导出前后模型输出是否一致

继承base实现通用onnxexporter：

核心实现：

- 由于模型设计规范是dict输出，而onnx不支持dict输出，需要对模型进行包装，转成单输出或者tuple输出。
- 可对模型进行simplify，优化模型架构，去除模型冗余结构（identity等等）

export_cli：命令行工具

## compression

目前支持和QAT以及针对onnx的PTQ（动态量化，静态量化）

### PTQ

- 动态量化：通用，会生成量化报告
- 静态量化：通用，会生成量化报告

### QAT

需配合模型设计，训练，层融合

- 模型设计：需插入量化桩，替换不支持量化的算子（concat，残差连接等等），如果存在无法量化也无法替换的算子，需插桩进行DQD
- 训练：需配置量化后端，加载预训练权重，QATHook，层融合策略等等
- 层融合：需继承FusionStrategy基类实现针对当前模型的融合策略，实现层融合，通常是（Linear，conv，relu，norm）

## interpretability

待实现

## 统一注册表

可避免繁复的导入问题，统一管理所有模块

实现了通用注册表类Registry

- 装饰器方式注册：register，可通过force参数避免重复注册问题
- 函数式注册：register_module，可通过force参数避免重复注册问题

MODELS：自定义的模型都必须注册

PREPROCESSINGS：自定义的preprocessing模块都必须注册

BACKBONES：自定义的backbone都必须注册

NECKS：自定义的neck模块都必须注册

LOSSES：自定义的损失函数都必须注册

OPTIMIZERS：自定义的优化器都必须注册

SCHEDULERS：自定义的调度策略都必须注册

HOOKS：自定义的钩子都必须注册

EVALUATORS：自定义的评估器都必须注册

INFERENCERS：自定义的推理器都必须注册

EXPORTERS：自定义的导出器都必须注册

MODEL_LOADERS：自定义的模型加载器都必须注册

FUSION_STRATEGIES：自定义的层融合策略都必须注册

注册方式示例：

```python
@MODELS.register('MyModel')
class MyModel(nn.Module):
    pass
```

以在FunFlow项目下的__init__内部实现自动注册逻辑

FunFlow下的所有模块都会被自动注册

local下的自定义模块也会被扫描检测并自动注册

## 统一日志系统

整个系统可以使用同一个日志系统，统一管理日志

- logger.debug - 调试信息，只写入文件
- logger.info - 普通信息，只写入文件
- logger.console - 重要信息，写入文件并显示在终端
- logger.warning - 警告信息，写入文件并显示在终端
- logger.error - 错误信息，写入文件并显示在终端

## 代码量

**总代码量：12,486 行**

| 目录 | 行数 |
|------|------|
| 根目录文件 | 545 行 |
| bin | 594 行 |
| compression | 2,010 行 |
| datasets | 317 行 |
| export | 821 行 |
| inference | 703 行 |
| models | 3,857 行 |
| trainer | 1,981 行 |
| utils | 1,658 行 |

## 下一步计划

- DDP支持（暂时没有多卡可以测试，所以暂时未实现）
- 可解释分析支持
- 剪枝流程支持
- 蒸馏流程支持
- 修复QAT逻辑