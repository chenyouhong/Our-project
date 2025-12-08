import pandas as pd

# 把这里的路径换成你刚刚下载好的 CSV 文件路径
csv_path = "data/alfa/test/carbonZ_2018-07-18-15-53-31_1_engine_failure/mavros-imu-data.csv"

try:
    df = pd.read_csv(csv_path, nrows=1)  # 只读第一行
    print("\n=== 你的 CSV 列名列表 ===")
    col_list = df.columns.tolist()
    print(col_list)

    print("\n=== 建议修改 p-gemini.py 中的 self.features 为: ===")
    # 自动帮你找匹配的列
    suggested = [c for c in col_list if "linear_acceleration" in c or "angular_velocity" in c]
    print(suggested)

except Exception as e:
    print(f"读取失败: {e}")