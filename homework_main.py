"""
基于ERA5数据和Aurora模型的合肥蜀山区气象预测
预测变量：2m温度、10m风场、平均海平面气压、大气温度/风场/比湿（多气压层）
空间范围：合肥蜀山区（北纬31.70°-32.00°，东经117.05°-117.40°）
时间设置：2025年11月4日（可自行修改，需保持日期一致性）
"""

from pathlib import Path
import cdsapi
import torch
import xarray as xr
import matplotlib.pyplot as plt
from aurora import Batch, Metadata
from aurora import Aurora, rollout  # 模型核心接口


# -------------------------- 1. 基础配置（需用户根据自身情况调整） --------------------------
# 1.1 蜀山区经纬度范围（[北, 西, 南, 东]，0.25°分辨率覆盖完整区域）
SHUSHAN_AREA = [32.00, 117.05, 31.70, 117.40]  # 关键：仅下载该区域数据，节省资源
# 1.2 数据下载路径（可自定义）
DOWNLOAD_PATH = Path("~/downloads/era5_shushan")
# 1.3 预测时间配置（输入数据时间：2025-11-04 00:00/06:00；预测步数：2步→12:00/18:00）
INPUT_YEAR = "2025"
INPUT_MONTH = "01"
INPUT_DAY = "01"
INPUT_TIMES = ["00:00", "06:00"]  # 输入数据的两个时间点
ROLLOUT_STEPS = 2  # 预测步数（每步6小时，2步对应12:00、18:00）
# 1.4 模型运行配置（True=Azure Foundry云运行，False=本地运行）
RUN_ON_FOUNDRY = False


# -------------------------- 2. 初始化CDS客户端与数据路径 --------------------------
# 扩展用户路径并创建文件夹（若不存在）
DOWNLOAD_PATH = DOWNLOAD_PATH.expanduser()
DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)

# 初始化CDS客户端（需提前配置~/.cdsapirc文件）
c = cdsapi.Client()


# -------------------------- 3. 下载ERA5数据（仅蜀山区范围） --------------------------
def download_era5_data():
    """下载静态变量、地表变量、大气变量（筛选蜀山区范围）"""
    # 3.1 下载静态变量（陆地掩码、土壤类型、位势，无时间维度）
    static_file = DOWNLOAD_PATH / "static.nc"
    if not static_file.exists():
        c.retrieve(
            dataset_name="reanalysis-era5-single-levels",
            request={
                "product_type": "reanalysis",
                "variable": [
                    "land_sea_mask",  # 陆地掩码（静态）
                    "soil_type",      # 土壤类型（静态）
                    "geopotential"    # 位势（静态）
                ],
                "year": INPUT_YEAR,
                "month": INPUT_MONTH,
                "day": INPUT_DAY,
                "time": INPUT_TIMES[0],  # 静态变量无时间差异，取任意时间点
                "area": SHUSHAN_AREA,    # 关键：筛选蜀山区范围
                "format": "netcdf"       # 数据格式（netCDF4）
            },
            target=str(static_file)
        )
    print(f"✅ 静态变量已保存至：{static_file}")

    # 3.2 下载地表变量（2m温度、10m风场、平均海平面气压）
    surf_file = DOWNLOAD_PATH / f"{INPUT_YEAR}-{INPUT_MONTH}-{INPUT_DAY}-surface-level.nc"
    if not surf_file.exists():
        c.retrieve(
            dataset_name="reanalysis-era5-single-levels",
            request={
                "product_type": "reanalysis",
                "variable": [
                    "2m_temperature",              # 2m温度
                    "10m_u_component_of_wind",     # 10m风场u分量
                    "10m_v_component_of_wind",     # 10m风场v分量
                    "mean_sea_level_pressure"      # 平均海平面气压
                ],
                "year": INPUT_YEAR,
                "month": INPUT_MONTH,
                "day": INPUT_DAY,
                "time": INPUT_TIMES,        # 输入时间点（00:00、06:00）
                "area": SHUSHAN_AREA,        # 筛选蜀山区范围
                "format": "netcdf"
            },
            target=str(surf_file)
        )
    print(f"✅ 地表变量已保存至：{surf_file}")

    # 3.3 下载大气变量（多气压层温度、风场、比湿、位势）
    atmos_file = DOWNLOAD_PATH / f"{INPUT_YEAR}-{INPUT_MONTH}-{INPUT_DAY}-atmospheric.nc"
    if not atmos_file.exists():
        c.retrieve(
            dataset_name="reanalysis-era5-pressure-levels",
            request={
                "product_type": "reanalysis",
                "variable": [
                    "temperature",              # 大气温度
                    "u_component_of_wind",      # 大气风场u分量
                    "v_component_of_wind",      # 大气风场v分量
                    "specific_humidity",        # 大气比湿
                    "geopotential"              # 大气位势
                ],
                "pressure_level": [           # 关键：选择需要的气压层（单位：hPa）
                    "50", "100", "150", "200", "250", "300",
                    "400", "500", "600", "700", "850", "925", "1000"
                ],
                "year": INPUT_YEAR,
                "month": INPUT_MONTH,
                "day": INPUT_DAY,
                "time": INPUT_TIMES,        # 输入时间点（00:00、06:00）
                "area": SHUSHAN_AREA,        # 筛选蜀山区范围
                "format": "netcdf"
            },
            target=str(atmos_file)
        )
    print(f"✅ 大气变量已保存至：{atmos_file}")

    return static_file, surf_file, atmos_file


# -------------------------- 4. 准备Aurora模型输入批次（Batch） --------------------------
def prepare_aurora_batch(static_file, surf_file, atmos_file):
    """将ERA5数据转换为Aurora模型所需的Batch格式"""
    # 4.1 读取netCDF数据（使用xarray，需netCDF4引擎）
    static_ds = xr.open_dataset(static_file, engine="netcdf4")
    surf_ds = xr.open_dataset(surf_file, engine="netcdf4")
    atmos_ds = xr.open_dataset(atmos_file, engine="netcdf4")

    # 4.2 构造地表变量字典（取前2个时间点：00:00、06:00，添加batch维度）
    surf_vars = {
        "t2m": torch.from_numpy(surf_ds["2m_temperature"].values[:2][None]),  # 2m温度
        "u10": torch.from_numpy(surf_ds["10m_u_component_of_wind"].values[:2][None]),  # 10m风u
        "v10": torch.from_numpy(surf_ds["10m_v_component_of_wind"].values[:2][None]),  # 10m风v
        "msl": torch.from_numpy(surf_ds["mean_sea_level_pressure"].values[:2][None])   # 平均海平面气压
    }

    # 4.3 构造静态变量字典（无时间维度，取第1个时间点）
    static_vars = {
        "z_static": torch.from_numpy(static_ds["geopotential"].values[0]),  # 静态位势
        "soil_type": torch.from_numpy(static_ds["soil_type"].values[0]),    # 土壤类型
        "lsm": torch.from_numpy(static_ds["land_sea_mask"].values[0])       # 陆地掩码
    }

    # 4.4 构造大气变量字典（取前2个时间点，添加batch维度）
    atmos_vars = {
        "t": torch.from_numpy(atmos_ds["temperature"].values[:2][None]),          # 大气温度
        "u": torch.from_numpy(atmos_ds["u_component_of_wind"].values[:2][None]),  # 大气风u
        "v": torch.from_numpy(atmos_ds["v_component_of_wind"].values[:2][None]),  # 大气风v
        "q": torch.from_numpy(atmos_ds["specific_humidity"].values[:2][None]),    # 大气比湿
        "z_atmos": torch.from_numpy(atmos_ds["geopotential"].values[:2][None])    # 大气位势
    }

    # 4.5 构造元数据（经纬度、时间、气压层）
    metadata = Metadata(
        lat=torch.from_numpy(surf_ds["latitude"].values),    # 纬度（蜀山区范围）
        lon=torch.from_numpy(surf_ds["longitude"].values),   # 经度（蜀山区范围）
        time=(surf_ds["valid_time"].values.astype("datetime64[s]").tolist()[1],),  # 参考时间（06:00）
        atmos_levels=tuple(int(level) for level in atmos_ds["pressure_level"].values)  # 气压层列表
    )

    # 4.6 生成Aurora Batch
    batch = Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=metadata
    )

    print("✅ Aurora模型输入批次准备完成")
    return batch, surf_ds  # 返回surf_ds用于后续可视化对比


# -------------------------- 5. 加载Aurora模型并执行预测 --------------------------
def run_aurora_prediction(batch):
    """加载预训练Aurora模型，执行预测（rollout）"""
    if not RUN_ON_FOUNDRY:
        # 本地运行：加载预训练模型（非LoRA版本）
        model = Aurora(use_lora=False)
        # 加载预训练权重（需联网，自动从microsoft/aurora下载）
        model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
        model.eval()  # 推理模式
        # 移动模型至GPU（若有），无GPU则自动使用CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        batch = batch.to(device)  # 输入批次同步至设备

        # 执行预测（关闭梯度计算，节省内存）
        with torch.inference_mode():
            preds = [pred.to("cpu") for pred in rollout(model, batch, steps=ROLLOUT_STEPS)]
        print(f"✅ 本地预测完成（设备：{device}），共{len(preds)}个时间步")

    else:
        # Azure Foundry云运行（需提前配置环境变量）
        import os
        import logging
        import warnings
        from aurora.foundry import BlobStorageChannel, FoundryClient, submit

        # 屏蔽警告，显示日志
        warnings.filterwarnings("ignore")
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
        logging.getLogger("aurora").setLevel(logging.INFO)

        # 初始化Foundry客户端（环境变量需提前设置）
        foundry_client = FoundryClient(
            endpoint=os.environ["FOUNDRY_ENDPOINT"],
            token=os.environ["FOUNDRY_TOKEN"]
        )
        # 初始化Blob存储通道
        channel = BlobStorageChannel(os.environ["BLOB_URL_WITH_SAS"])

        # 提交预测任务
        preds = list(submit(
            batch=batch,
            model_name="aurora-0.25-pretrained",
            num_steps=ROLLOUT_STEPS,
            foundry_client=foundry_client,
            channel=channel
        ))
        print("✅ Azure Foundry云预测完成")

    return preds


# -------------------------- 6. 可视化预测结果（对比ERA5输入） --------------------------
def visualize_results(preds, surf_ds):
    """可视化2m温度预测结果（蜀山区），对比ERA5输入数据"""
    # 转换温度单位：Kelvin → Celsius（减去273.15）
    def kelvin_to_celsius(kelvin_data):
        return kelvin_data - 273.15

    # 初始化画布（2行2列：2个预测时间步 × 预测/ERA5对比）
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("合肥蜀山区2m温度预测（Aurora模型）", fontsize=16, fontweight="bold")

    # 预测时间标签（输入00/06 → 预测12/18）
    pred_times = ["2025-11-04 12:00", "2025-11-04 18:00"]

    # 遍历2个预测时间步
    for i in range(ROLLOUT_STEPS):
        pred = preds[i]
        # 1. 绘制预测结果（左列）
        pred_t2m = kelvin_to_celsius(pred.surf_vars["t2m"][0].numpy())  # 去除batch维度
        im1 = axs[i, 0].imshow(
            pred_t2m,
            cmap="coolwarm",
            vmin=-5,  # 温度范围（根据合肥1月气候调整）
            vmax=15,
            origin="upper"  # 纬度从上到下递减（符合地理坐标）
        )
        axs[i, 0].set_title(f"预测结果：{pred_times[i]}", fontweight="bold")
        axs[i, 0].set_ylabel("纬度（°N）")
        axs[i, 0].set_xticks([])  # 简化显示，隐藏经纬度刻度
        axs[i, 0].set_yticks([])

        # 2. 绘制ERA5输入参考（右列，取输入的最后一个时间点06:00）
        era5_t2m = kelvin_to_celsius(surf_ds["2m_temperature"][1].values)  # 06:00数据
        im2 = axs[i, 1].imshow(
            era5_t2m,
            cmap="coolwarm",
            vmin=-5,
            vmax=15,
            origin="upper"
        )
        axs[i, 1].set_title(f"ERA5参考：2025-11-04 06:00", fontweight="bold")
        axs[i, 1].set_xticks([])
        axs[i, 1].set_yticks([])

    # 添加颜色条（统一尺度）
    cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), shrink=0.8, pad=0.02)
    cbar.set_label("2m温度（°C）", fontsize=12)

    # 调整布局并保存
    plt.tight_layout()
    save_path = DOWNLOAD_PATH / "shushan_aurora_prediction.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 预测结果图已保存至：{save_path}")


# -------------------------- 主函数（串联所有步骤） --------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("开始合肥蜀山区气象预测（基于ERA5 + Aurora）")
    print("=" * 60)

    # 步骤1：下载蜀山区ERA5数据
    static_file, surf_file, atmos_file = download_era5_data()

    # 步骤2：准备Aurora输入批次
    batch, surf_ds = prepare_aurora_batch(static_file, surf_file, atmos_file)

    # 步骤3：运行模型预测
    preds = run_aurora_prediction(batch)

    # 步骤4：可视化结果
    visualize_results(preds, surf_ds)

    print("=" * 60)
    print("预测流程全部完成！")
    print("=" * 60)