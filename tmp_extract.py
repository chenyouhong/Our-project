from pathlib import Path
import pandas as pd

SRC_TRAIN = Path('data/alfa/train_nominal')
DST_TRAIN = Path('data/alfa/train')
SRC_TEST = Path('data/alfa/test_failure')
DST_TEST = Path('data/alfa/test')


def rename_columns(df):
    mapping = {
        '%time': 'Time',
        'field.header.seq': 'header.seq',
        'field.header.stamp': 'header.stamp',
        'field.header.frame_id': 'header.frame_id',
    }

    def _rename(col):
        if col in mapping:
            return mapping[col]
        if col.startswith('field.orientation_covariance'):
            return 'orientation_covariance_' + col.split('covariance')[-1]
        if col.startswith('field.angular_velocity_covariance'):
            return 'angular_velocity_covariance_' + col.split('covariance')[-1]
        if col.startswith('field.linear_acceleration_covariance'):
            return 'linear_acceleration_covariance_' + col.split('covariance')[-1]
        if col.startswith('field.'):
            return col[len('field.'):]
        return col

    return df.rename(columns=_rename)


def convert_file(src_csv: Path, dst_csv: Path):
    df = pd.read_csv(src_csv)
    df = rename_columns(df)

    if 'Time' not in df or 'header.stamp' not in df:
        raise RuntimeError(f'Missing required columns in {src_csv}')

    df['Time'] = df['Time'] / 1_000_000_000.0

    stamp = df.pop('header.stamp').astype('int64')
    df.insert(2, 'header.stamp.secs', stamp // 1_000_000_000)
    df.insert(3, 'header.stamp.nsecs', stamp % 1_000_000_000)

    for col in ['header.seq', 'header.stamp.secs', 'header.stamp.nsecs']:
        if col in df.columns:
            df[col] = df[col].astype('int64')

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_csv, index=False)
    return len(df)


def find_imu_file(src_dir: Path):
    candidates = sorted(src_dir.glob('*mavros-imu-data*.csv'))
    if not candidates:
        return None
    for c in candidates:
        name = c.name.lower()
        if 'raw' not in name:
            return c
    return candidates[0]


def convert_missing(src_root: Path, dst_root: Path):
    missing = [p for p in src_root.iterdir() if p.is_dir() and not (dst_root / p.name).exists()]
    results = []
    for src_dir in missing:
        imu_file = find_imu_file(src_dir)
        if imu_file is None:
            print(f'[skip] No imu csv in {src_dir}')
            continue
        dst_dir = dst_root / src_dir.name
        dst_csv = dst_dir / 'mavros-imu-data.csv'
        n_rows = convert_file(imu_file, dst_csv)
        results.append((src_dir.name, n_rows))
        print(f'[ok] {src_dir.name}: {imu_file.name} -> {dst_csv} ({n_rows} rows)')
    return results


if __name__ == '__main__':
    print('Converting missing train_nominal -> train...')
    convert_missing(SRC_TRAIN, DST_TRAIN)
    print('Converting missing test_failure -> test...')
    convert_missing(SRC_TEST, DST_TEST)
