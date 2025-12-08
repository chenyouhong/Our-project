import glob
import os
import glob

def get_flight_id(path: str) -> str:
    """
    例：
    data/alfa/train/carbonZ_2018-07-18-15-53-31_1_engineFullPowerLoss/mavros-imu-data.csv
    -> 2018-07-18-15-53-31_1
    """
    dirname = os.path.basename(os.path.dirname(path))
    # 如果有 carbonZ_ 前缀就去掉（根据你的实际情况改）
    if dirname.startswith("carbonZ_"):
        dirname = dirname[len("carbonZ_"):]
    parts = dirname.split("_")
    # ['2018-07-18-15-53-31', '1', 'engineFullPowerLoss']
    if len(parts) >= 2 and parts[1].isdigit():
        return parts[0] + "_" + parts[1]
    else:
        # 例如 2018-07-30-16-29-45_engine...
        return parts[0]

train_files = glob.glob('data/alfa/train/**/mavros-imu-data.csv', recursive=True)

print("共找到训练文件数:", len(train_files))
print("前 5 个路径与解析出的 flight_id:")

for p in train_files[:5]:
    print(p, "  -->  ", get_flight_id(p))

train_files = glob.glob('data/alfa/train/**/mavros-imu-data.csv', recursive=True)

print("共找到训练文件数:", len(train_files))
print("前 5 个路径示例:")
for p in train_files[:5]:
    print(p)
