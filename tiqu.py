import bagpy
from bagpy import bagreader
import pandas as pd

# 读取 bag 文件
b = bagreader(
    'data/alfa/test_failure/carbonZ_2018-10-05-14-34-20_2_right_aileron_failure_with_emr_traj/carbonZ_2018-10-05-14-34-20_2_right_aileron_failure_with_emr_traj.bag')

# 查看所有 Topic
print(b.topic_table)

# 提取 IMU 数据 (这会自动生成一个 CSV 文件)
imu_csv = b.message_by_topic('/mavros/imu/data')
print(f"IMU CSV saved at: {imu_csv}")