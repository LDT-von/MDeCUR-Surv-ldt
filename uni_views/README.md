# 同一 patch 的 UNI 双视图特征

本工具只处理明确列出的 WSI 和已有坐标：读取同一 patch 一次，在图像像素上分别施加两个完整视野的 D4 变换（0/90/180/270 度旋转，可附加镜像），再用同一个冻结 UNI v1 提取特征。每个视图保留相同 patch 数、行顺序和坐标。不重新分割组织、不生成患者划分、不随机删除 patch、不把特征向量当图片翻转、不保存海量 patch 图片。与旧 SlotSPE 代码无依赖。

这是工程实现；图像增强是否改善患者生存预测、稳定性权重是否有用，均未验证。两个固定缓存视图也不等于每个 epoch 重新随机增强；UNI 可能已经对这些方向变化不敏感，必须检查患者权重的分布与对照结果。

## 1. 文件和环境

入口：仓库根目录 `extract_uni_views.py`，子命令 `plan`、`doctor`、`extract`。

清单每行一张切片，必需字段：

```text
patient_id,slide_id,wsi_path,coords_h5
```

复制 `slides.csv.example` 后填真实路径，文件名改为 `slides.csv`。示例中的 `FAKE_*` 不能用于实际运行。路径相对于清单目录，支持绝对路径。`slide_id` 必须与 WSI 和坐标文件去掉扩展名后的完整名字一致；不能只保留病例短 ID。一个患者可以列多张切片，但同一 WSI/坐标文件不能重复出现。患者归属来自现有元数据，不能按目录数量猜测或重新分组。

坐标 H5 必须包含整数 `coords[N,2]`，坐标顺序为 level-0 `(x,y)`；`coords.attrs` 必须有 `patch_size`、`patch_level=0`。本版明确只支持 level 0，不猜倍率、不自动转换其它 level。若有 `level_dim`，必须与实际 WSI 的 level-0 尺寸相同。`doctor` 检查报告记录实际 MPP、物镜倍率元数据，不自动声称所有 512px 都等于某个物理尺度。

环境需要 numpy、h5py、Pillow、torch、torchvision、timm、openslide-python 和 OpenSlide 原生库。本机不会自动安装任何依赖。依赖清单见 `requirements-extraction.txt`；PyTorch/torchvision 应单独按目标 GPU 选择，在隔离环境中安装。旧 PyTorch 若无 `weights_only=True` 会拒绝加载，不退回不安全的 pickle。GPU 的实际 kernel/显存可用性仍需小规模试跑验证，doctor 只报告 CUDA 和架构信息。

UNI 权重必须是本地的 **UNI v1 ViT-L/16、1024 维**原始 state_dict，不支持把 UNI2-h 权重、已提取的 `.pt` 特征或生存模型 checkpoint 当编码器权重。严格加载所有参数，`eval()`、冻结梯度、`inference_mode()`，不训练 UNI、不下载权重。

模型构造及参考归一化依据 [UNI 官方 v1 本地加载示例](https://github.com/mahmoodlab/UNI/blob/main/README_old.md#2-downloading-weights--creating-model)。仓库附带 `recipe_uni_v1.json` 使用完整方形 patch 直接双线性缩放到 224，再按 ImageNet 均值/标准差归一化。它是明确的新提取配方，**不代表已证明等同于下载特征的历史配方**。若确有先缩到 256 再到 224 的来源证据，可另存配方 JSON，将 `resize_sizes` 改为 `[256,224]`；依然不裁剪内容。

## 2. 先检查，不跑 UNI

以下命令都从仓库根目录执行，`PATH_TO_*` 必须替换。

```powershell
python extract_uni_views.py plan --manifest slides.csv --recipe uni_views/recipe_uni_v1.json
python extract_uni_views.py doctor --manifest slides.csv --recipe uni_views/recipe_uni_v1.json --checkpoint "PATH_TO_LOCAL_UNI_V1/pytorch_model.bin" --device cuda:0
```

- `plan`：只读清单、坐标和文件元信息；不打开 WSI、不读取像素、不加载模型、不创建输出目录。汇总所有缺失输入并非零退出；不静默排除缺 WSI 的切片。报告完整 float32 特征字节数。
- `doctor`：在上述基础上打开 WSI 头信息，核对实际尺寸、边缘坐标和环境；不读取 patch 像素、不构建/加载 UNI、不修改文件。返回码 0 才代表这些检查通过；不是实际推理成功证明。
- 21 张缺 WSI 的切片、对应患者及既有 splits 必须由数据协议明确处理，工具不补零、不删除患者、不改划分。

边缘默认策略 `--edge-policy error`，存在边界不完整 patch 就拒绝提取。若核查后决定保留并用白色补齐，可在 doctor/extract 中明确加 `--edge-policy pad-white`。读取结果的透明像素统一合成白底。不能直接跳过边缘行，否则会破坏行对应关系。起点完全落在 WSI 外、负坐标或记录尺寸与实际不符始终报错，即使选择白色补齐也不放行。

## 3. 小范围测时，再决定全量

先单独准备仅含少量代表性切片的 `slides_pilot.csv`。以下是待用户在目标电脑执行的试提取格式，不是已经执行的患者任务：

```powershell
python extract_uni_views.py extract --manifest slides_pilot.csv --recipe uni_views/recipe_uni_v1.json --checkpoint "PATH_TO_LOCAL_UNI_V1/pytorch_model.bin" --out uni_view_outputs/pilot --device cuda:0 --batch-size 16 --limit-patches 1000 --include-reference
```

默认按 float32 计算并以 float32 保存。`--batch-size 16` 指 16 个原 patch，每批实际为 32 个增强图像；加参考则为 48 个图像。显存不足需使用新的输出目录减小 batch size。可明确选 `--amp float16` 或 `--amp bfloat16`，计算精度写入运行记录，保存仍是 float32；需要单独检验数值差异，不自动开启。

`--limit-patches` 是测时/排查用的前 K 行，不保证代表全部组织分布；标记为 pilot，训练加载器拒绝把这种不完整切片当完整患者 bag。选择的切片仍会检查全部坐标几何。更改行数、精度、seed、配方或 batch size 后不能直接 resume 原运行。

每张切片记录 `elapsed_seconds`、`patches_per_second`、`images_per_second`；时间包含读 patch、增强、UNI 前向与特征写入，另有整次运行 wall time。2/3 视图的测时不能混用：总耗时近似 `总patch数 × 视图数 / 实测images_per_second`，另加加载、校验开销。WSI 读盘速度和组织区域不同，少量测试不能保证全量速度。

两个 D4 变换是按 seed、完整 slide_id、原始行号独立确定的，同一 patch 偶然抽到相同变换是正常的（约 1/8），会记录同变换对数。改变 batch size 不改变图像视图。它们不是两种模型的特征，也不是额外患者。

全量命令移除 `--limit-patches`，使用完整的显式切片清单与**新的**输出目录。`--include-reference` 是否开启必须在运行前确定。双 GPU 可使用两个事先核对且不相交的切片清单、两个输出目录和 `cuda:0`/`cuda:1` 分别运行；工具不自动启动后台任务或跨 GPU 分发。

## 4. 输出、存储和续跑

每张切片保存一个 `<slide_id>.h5`，包含两份特征数据集：

```text
coords                [N,2]       原坐标，原顺序
row_index             [N]         0..N-1
transform_a/b         [N]         0..7 的 D4 变换码
features_a/b          [N,1024]    两份 float32 UNI 特征
features_reference    [N,1024]    仅 --include-reference 时，未增强参考
```

属性记录输入路径、WSI 大小/mtime、原坐标文件及数组 SHA256、UNI 权重 SHA256、预处理配方、seed、边缘策略、原始/实际行数、完成状态、各特征数据集 SHA256。`run.json` 还记录清单哈希、运行环境、实现源码哈希和运行参数。WSI 使用大小/mtime 指纹，不是全文件内容哈希；这些记录不能替代临床来源审查。

默认只存两份增强特征，新增 payload 为 `N×1024×4×2` 字节。按照报告的整套 457 张单份约 28.79GB，约新增 **57.58GB**。若加未增强参考则约新增 **86.37GB**（三份），不能继续按两份估空间；436 张交集的准确数字由 plan 对真实 N 汇总。HDF5 元数据和坐标还有少量开销；中断残留也需要空间。程序检查剩余磁盘加 10% 和 64MiB 余量，但估计不包含其它进程同时写盘。

输出目录必须新建且位于源 WSI/坐标目录之外。不会覆盖原始特征、WSI、坐标或现有完整输出。每张切片先写随机后缀 `.partial.*`，完整写完后才发布为 `.h5`，异常时不会产生伪完成文件。发布使用同盘原子硬链接，避免并发写入时覆盖已有文件；输出盘应支持硬链接（例如 NTFS、ext4）。不支持的文件系统会报错并保留临时产物，不退回覆盖式写入。

续跑：完全相同的命令追加 `--resume`。程序核对运行合同，并逐块校验已完成文件的行索引、坐标、变换和特征哈希；仅重做尚未完成的整张切片，不是从某个 patch 恢复。残留 `.partial.*` 不会自动删除。运行锁 `.extract.lock` 防止同目录并发写入；进程被强制结束可能留下锁，需确认 PID 已退出再人工处理锁，不能盲目清除正在使用的锁。

## 5. 接入当前癌症模型

新模式为 `--path-views cached_uni`。旧的 `legacy_patch_dropout` 留作历史合成实验对照，不自动冒充图像增强。

为避免只凭“行数一样”就把下载特征与新特征混在一起，新加载器**要求 `--include-reference` 输出**：主生存损失和验证/测试使用同文件的未增强 reference；仅训练患者加载 A/B 做辅助约束。这样三份特征的坐标和预处理来自同一次受控流程，不声称与外部缓存逐行等价。未开启 reference 的两视图输出可以保留，但当前严格训练接口拒绝直接混用旧 `.pt`。

患者清单使用 `patients.csv.example` 的字段；每行一个唯一患者，沿用既有划分、RNA、时间和事件。`wsi_h5` 可以是一个 H5 路径，或 CSV 中正确转义的 JSON 路径列表，例如该字段内容为 `["slideA.h5","slideB.h5"]`，按列表顺序拼接同一患者的多个切片；所有文件的 patient_id 必须一致。加载器检查完整行、特征哈希、重复切片、跨划分复用、编码器权重和预处理一致性。

```powershell
python train_cancer.py --manifest patients.csv --out runs/NEW_CACHED_VIEWS_RUN --starts 0 1 3 --horizon 3 --time-unit years --endpoint "OS from an explicitly defined baseline" --mode stability --path-views cached_uni --rna-noise 0.05
```

以上生存参数只是接口示例，需要遵循真实终点/时间单位方案。新模式强制 `patch_keep=1`；与输入相同的 mask 应用于两份视图，不做第二次 patch 抽样。RNA 仍是原实现的特征噪声，`--rna-noise` 可显式设置，这不是已验证的基因增强方法。

**当前训练器仍为全批次 padding 的研究实现。新特征提取器是分批流式的，但这不等于训练器已经能承载数百万 patch。** 加载三份特征后，RAM 和 padding 需求可能很高；在目标项目集成可扩展训练前不要直接把整个 BLCA 队列塞入此示例训练器。本次没有修改 Tri-Surv 的 resnet50 硬编码，也没有声称其原 loader 可直接读取本输出。

## 6. 本地验证边界

```powershell
python -B -m unittest -v test_uni_views test_cancer_model test_cancer_data test_cancer_metrics test_prototype test_pipeline
```

新检查使用合成像素、伪 WSI 读取器、伪编码器和临时文件，覆盖 D4 像素保留、一次读取、批次无关的逐行对应、边缘处理、输入变更、损坏文件、原子发布、续跑、pilot 拒绝、患者归属、冻结编码器包装和完整训练接口。它们证明接口行为，不证明真实 SVS 解码、本机 UNI 权重兼容性、GPU kernel、目标机器速度或患者预测效果。实际 UNI/WSI 小范围验证尚待目标电脑执行。
