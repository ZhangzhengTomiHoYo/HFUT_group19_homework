1. 项目介绍
合肥工业大学研究生二年级深度学习课程编程作业，第19组作业。
本项目基于ERA5 再分析数据和Microsoft Aurora 预训练模型，实现对合肥蜀山区的气象变量预测。核心预测变量包括：
近地表变量：2m 温度、10m 风场（u/v 分量）、平均海平面气压
大气垂直变量：多气压层（50~1000hPa）温度、风场、比湿、位势
预测时间逻辑：输入 2023 年 1 月 1 日 00:00/06:00 的 ERA5 数据，通过 Aurora 模型滚动预测（2 步），输出当日 12:00/18:00 的气象结果，并可视化 2m 温度对比。
2. 环境准备
2.1 基础环境要求
Python 3.8~3.11（推荐 3.10，避免版本兼容问题）
内存 ≥ 16GB（若使用 GPU，显存 ≥ 8GB）
网络连接（用于下载 ERA5 数据和 Aurora 预训练权重）
2.2 依赖包安装
打开终端，执行以下命令安装所有必需依赖：
bash
# 核心依赖（CDS数据下载、模型、数据处理、可视化）
pip install cdsapi matplotlib torch==2.1.0 xarray netCDF4
# Aurora模型（需从官方源安装，确保版本匹配）
pip install git+https://github.com/microsoft/aurora.git@main
2.3 CDS 账户配置（关键步骤）
ERA5 数据需从Copernicus Climate Data Store（CDS） 下载，需提前注册并配置 API：
注册 CDS 账户：访问CDS 官网，完成注册（免费）。
获取 API 密钥：登录后，进入用户设置页，复制页面中的 “API key”（格式如：12345:abcdef12-3456-7890-abcd-ef1234567890）。
创建配置文件：
Windows：在C:\Users\你的用户名\.cdsapirc创建文件。
Linux/Mac：在~/.cdsapirc创建文件。
文件内容（替换<API key>为你的密钥）：
plaintext
url: https://cds.climate.copernicus.eu/api/v2
key: <API key>
接受 ERA5 使用条款：访问ERA5 单层级数据集页和ERA5 气压层数据集页，点击 “Accept Terms”（不接受无法下载数据）。
3. 代码使用步骤
3.1 调整配置参数（可选）
打开Hefei_Shushan_Aurora_Pred.py，根据需求修改以下参数：
SHUSHAN_AREA：蜀山区经纬度范围（默认已覆盖完整区域，无需修改）。
DOWNLOAD_PATH：数据下载路径（默认~/downloads/era5_shushan，可自定义）。
INPUT_YEAR/MONTH/DAY：输入数据日期（默认 2023-01-01，需保持年月日一致性）。
ROLLOUT_STEPS：预测步数（默认 2 步→12:00/18:00，每步 6 小时）。
RUN_ON_FOUNDRY：是否使用 Azure 云运行（默认 False，本地运行）。
3.2 运行代码
在终端进入代码所在目录，执行：
bash
python Hefei_Shushan_Aurora_Pred.py
3.3 输出文件
运行完成后，在DOWNLOAD_PATH目录下生成 3 类文件：
数据文件：static.nc（静态变量）、2023-01-01-surface-level.nc（地表变量）、2023-01-01-atmospheric.nc（大气变量）。
可视化结果：shushan_aurora_prediction.png（2m 温度预测对比图）。
4. 关键模块说明
模块	功能描述
数据下载（CDS API）	筛选蜀山区范围的 ERA5 数据，避免下载全球数据（节省时间和磁盘空间）。
Batch 准备	将 ERA5 的 netCDF 数据转换为 Aurora 模型所需的 Tensor 格式，添加 batch 维度。
模型加载与预测	本地加载预训练 Aurora 模型（非 LoRA 版本），支持 GPU 加速，执行滚动预测。
可视化	对比展示 2m 温度预测结果与 ERA5 输入参考，使用coolwarm色表直观呈现温度分布。
5. 常见问题解决
5.1 CDS API 下载报错
错误 1：Permission denied: 403 → 未接受 ERA5 数据集的使用条款，需前往 CDS 数据集页点击 “Accept Terms”。
错误 2：API key not found → .cdsapirc文件路径或内容错误，检查文件位置和 key 格式。
错误 3：Connection timeout → 网络不稳定，可重试或使用代理（需在.cdsapirc中添加proxy: http://代理地址:端口）。
5.2 模型权重下载失败
问题：load_checkpoint时报错 “无法连接 microsoft/aurora” → 网络问题或 GitHub 访问受限。
解决：手动下载权重文件（aurora-0.25-pretrained.ckpt），并修改代码中load_checkpoint路径为本地路径：
python
运行
model.load_checkpoint("本地权重文件路径/aurora-0.25-pretrained.ckpt")
5.3 GPU 内存不足
问题：CUDA out of memory → GPU 显存不足。
解决：
改用 CPU 运行（代码会自动检测，无需修改）。
减小预测区域（调整SHUSHAN_AREA范围，仅保留核心区域）。
降低ROLLOUT_STEPS（减少预测步数）。
6. 注意事项
数据版权：ERA5 数据属于 Copernicus Climate Change Service（C3S），仅可用于非商业研究，需遵守CDS 数据使用政策。
模型适用范围：Aurora 模型预训练版本适用于全球 0.25° 分辨率，本项目通过区域筛选适配蜀山区，预测精度受输入数据和模型本身限制，仅供参考。
时间一致性：输入数据的year/month/day需保持一致，否则会导致 xarray 读取变量时维度不匹配。
7. 扩展建议
预测变量扩展：在surf_vars和atmos_vars中添加更多变量（如降水量、云量），需先在 CDS 下载请求中添加对应变量名（参考ERA5 变量列表）。
时间序列预测：修改INPUT_TIMES为更长的时间序列（如 1 天内的所有 6 小时间隔），增加ROLLOUT_STEPS实现多日预测。
结果量化分析：添加 RMSE、MAE 等指标计算，对比预测结果与 ERA5 后续时间点的观测数据（需下载对应时间的 ERA5 数据）