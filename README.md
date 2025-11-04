# 合肥蜀山区气象预测项目（基于Fork微软开源Aurora仓库）
本项目基于Fork微软开源的Aurora仓库（https://github.com/microsoft/aurora）开发，结合1.pdf文档中ERA5数据处理与Aurora模型使用逻辑，实现对合肥蜀山区的气象变量预测，以下为项目完整说明。


## 1. 项目介绍
本项目为合肥工业大学研究生二年级深度学习课程编程作业（第19组作业），基于ERA5再分析数据和Microsoft Aurora预训练模型，聚焦合肥蜀山区气象预测，核心内容如下：
- **预测变量**：
  - 近地表变量：2m温度、10m风场（u/v分量）、平均海平面气压
  - 大气垂直变量：多气压层（50~1000hPa）温度、风场、比湿、位势
- **时间逻辑**：输入2023年1月1日00:00/06:00的ERA5数据，通过Aurora模型执行2步滚动预测，输出当日12:00/18:00的气象结果，并可视化2m温度与ERA5输入数据的对比关系。


## 2. 环境准备
### 2.1 基础环境要求
- Python版本：3.8~3.11（推荐3.10，避免版本兼容问题）
- 硬件配置：内存≥16GB；若使用GPU，显存≥8GB
- 网络要求：需联网（用于下载ERA5数据和Aurora预训练权重）

### 2.2 依赖包安装
打开终端，执行以下命令安装所有必需依赖，其中Aurora模型直接基于微软开源仓库安装，确保与1.pdf中“非微调版本Aurora”使用要求一致🔶1-7：
```bash
# 核心依赖（CDS数据下载、数据处理、可视化、深度学习框架）
pip install cdsapi matplotlib torch==2.1.0 xarray netCDF4
# 安装微软开源Aurora模型（指定main分支，确保版本匹配）
pip install git+https://github.com/microsoft/aurora.git@main
```

### 2.3 CDS账户配置（关键步骤）
ERA5数据需从Copernicus Climate Data Store（CDS）下载，需按以下步骤完成配置（1.pdf中明确要求注册账户与接受条款）🔶1-10：
1. **注册CDS账户**：访问[CDS官网](https://cds.climate.copernicus.eu/)，完成免费注册。
2. **获取API密钥**：登录后进入[用户设置页](https://cds.climate.copernicus.eu/user/settings)，复制“API key”（格式示例：`12345:abcdef12-3456-7890-abcd-ef1234567890`）。
3. **创建配置文件**：
   - Windows系统：在`C:\Users\你的用户名\.cdsapirc`路径下创建文件
   - Linux/Mac系统：在`~/.cdsapirc`路径下创建文件
   - 文件内容（替换`<API key>`为你的实际密钥）：
     ```plaintext
     url: https://cds.climate.copernicus.eu/api/v2
     key: <API key>
     ```
4. **接受使用条款**：分别访问[ERA5单层级数据集页](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels)和[ERA5气压层数据集页](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels)，点击“Accept Terms”（不接受将无法下载数据）。


## 3. 代码使用步骤
### 3.1 调整配置参数（可选）
打开核心代码文件`Hefei_Shushan_Aurora_Pred.py`，可根据需求修改以下关键参数（参数逻辑与1.pdf中数据下载、模型运行配置一致）🔶1-14：
- `SHUSHAN_AREA`：蜀山区经纬度范围（默认`[32.00, 117.05, 31.70, 117.40]`，已覆盖完整区域，无需修改）
- `DOWNLOAD_PATH`：数据下载路径（默认`~/downloads/era5_shushan`，可自定义本地路径）
- `INPUT_YEAR/MONTH/DAY`：输入数据日期（默认2023-01-01，需保持年月日一致性）
- `ROLLOUT_STEPS`：预测步数（默认2步，对应输出12:00/18:00结果，每步间隔6小时）
- `RUN_ON_FOUNDRY`：是否使用Azure云运行（默认`False`，即本地运行；若需云运行需额外配置环境变量）

### 3.2 运行代码
在终端中进入代码所在目录，执行以下命令启动预测流程：
```bash
python Hefei_Shushan_Aurora_Pred.py
```

### 3.3 输出文件
运行完成后，在`DOWNLOAD_PATH`指定的路径下生成3类文件（文件类型与1.pdf中数据下载、可视化输出逻辑一致）🔶1-19🔶1-20🔶1-21：
- **数据文件**：
  - `static.nc`：静态变量（陆地掩码、土壤类型、位势）
  - `2023-01-01-surface-level.nc`：地表变量（2m温度、10m风场、平均海平面气压）
  - `2023-01-01-atmospheric.nc`：大气变量（多气压层温度、风场、比湿、位势）
- **可视化结果**：
  - `shushan_aurora_prediction.png`：2m温度预测对比图（左列Aurora预测结果，右列ERA5参考数据）


## 4. 关键模块说明
| 模块                | 功能描述                                                                 | 关联1.pdf对应逻辑               |
|---------------------|--------------------------------------------------------------------------|--------------------------------|
| 数据下载（CDS API） | 筛选蜀山区范围的ERA5数据，避免下载全球数据，节省时间与磁盘空间           | 🔶1-14🔶1-15（数据下载流程） |
| Batch准备           | 将ERA5的netCDF数据转换为Aurora模型所需的Tensor格式，添加batch维度        | 🔶1-22🔶1-23（批次准备步骤） |
| 模型加载与预测      | 本地加载预训练Aurora模型（非LoRA版本），支持GPU加速，执行滚动预测        | 🔶1-25🔶1-26（模型运行逻辑） |
| 可视化              | 对比展示2m温度预测结果与ERA5输入参考，用`coolwarm`色表呈现温度分布       | （可视化代码框架）        |


## 5. 常见问题解决
### 5.1 CDS API下载报错
- **错误1：Permission denied: 403**  
  原因：未接受ERA5数据集使用条款  
  解决：前往1.3中提及的ERA5单层级/气压层数据集页，点击“Accept Terms”。
- **错误2：API key not found**  
  原因：`.cdsapirc`文件路径错误或密钥格式有误  
  解决：检查文件是否在指定路径（Windows：`C:\Users\用户名\.cdsapirc`；Linux/Mac：`~/.cdsapirc`），确认密钥格式与官网一致。
- **错误3：Connection timeout**  
  原因：网络不稳定或访问受限  
  解决：重试下载；若需代理，在`.cdsapirc`中添加`proxy: http://代理地址:端口`。

### 5.2 模型权重下载失败
- **问题**：`load_checkpoint`报错“无法连接microsoft/aurora”  
  原因：网络问题或GitHub访问受限  
  解决：手动下载权重文件[aurora-0.25-pretrained.ckpt](https://huggingface.co/microsoft/aurora/resolve/main/aurora-0.25-pretrained.ckpt)，修改代码中权重加载路径：
  ```python
  # 原代码：model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
  # 修改后（替换为本地权重路径）：
  model.load_checkpoint("本地权重文件路径/aurora-0.25-pretrained.ckpt")
  ```

### 5.3 GPU内存不足
- **问题**：`CUDA out of memory`  
  原因：GPU显存不足以承载模型与数据  
  解决：
  1. 自动切换CPU运行（代码会检测设备，无需修改）；
  2. 缩小`SHUSHAN_AREA`范围，减少输入数据维度；
  3. 降低`ROLLOUT_STEPS`，减少预测步数。


## 6. 注意事项
1. **数据版权**：ERA5数据属于Copernicus Climate Change Service（C3S），仅可用于非商业研究，需遵守[CDS数据使用政策](https://cds.climate.copernicus.eu/cdsapp/#/terms/licence-to-use-copernicus-climate-change-service-information)。
2. **模型适用范围**：Aurora预训练模型默认适配全球0.25°分辨率，本项目通过区域筛选适配蜀山区，预测精度受输入数据与模型本身限制，仅供课程作业参考。
3. **时间一致性**：输入数据的`year/month/day`需保持一致（如均为2023-01-01），否则会导致xarray读取变量时维度不匹配🔶1-15。


## 7. 扩展建议
1. **预测变量扩展**：在代码的`surf_vars`（地表变量）和`atmos_vars`（大气变量）字典中添加更多变量（如降水量、云量），需先在CDS数据下载请求中添加对应变量名（参考[ERA5变量列表](https://confluence.ecmwf.int/display/CKB/ERA5+variable+list)）。
2. **时间序列预测**：修改`INPUT_TIMES`为1天内所有6小时间隔（如`["00:00", "06:00", "12:00", "18:00"]`），增加`ROLLOUT_STEPS`实现多日连续预测。
3. **结果量化分析**：添加RMSE（均方根误差）、MAE（平均绝对误差）等指标计算代码，对比Aurora预测结果与ERA5后续时间点观测数据（需额外下载对应时间的ERA5数据）。