from pathlib import Path

import pandas as pd


def parse_path_name(some_path: Path) -> dict:
    plate, well, sample_str, name = str(some_path.stem).split("_")
    n_sample = int(sample_str[1:])
    channel = 1 if name.startswith("w1") else 2

    return {"plate": plate, "well": well, "sample": n_sample, "channel": channel}


# def extract_meta(data_root: Path, desc_path: Path, save_path: Path):
def extract_meta(data_root: Path, desc_path: Path) -> pd.DataFrame:
    img_paths = data_root.glob("**/*.tif")
    img_paths = list(img_paths)

    no_thumbnail_paths = [p for p in img_paths if "thumb" not in p.stem]
    meta_df_lines = []
    for img_path in no_thumbnail_paths:
        meta = parse_path_name(img_path)
        meta.update({"path": str(img_path.relative_to(data_root))})
        meta_df_lines.append(meta)

    meta_df = pd.DataFrame(meta_df_lines)
    desc_df = pd.read_csv(desc_path)
    merged_meta = pd.merge(meta_df, desc_df,
                           left_on='well', right_on='384-WELL').drop(["384-WELL"], axis=1)
    sorted_merged_meta = merged_meta.sort_values(["plate", "well", "sample"])

    return sorted_merged_meta
    # sorted_merged_meta.to_csv(save_path, index=None)


def extract_dataset_split(images_meta: pd.DataFrame) -> pd.DataFrame:
    data_meta_lines = []

    for idx, row in images_meta.groupby(["plate", "well", "sample"]):
        two_channels = row.sort_values('channel')
        meta_line = {"plate": two_channels.plate.any(),
                     "channel_A": two_channels.iloc[0].path,
                     "channel_B": two_channels.iloc[1].path,
                     "compound": two_channels["compound (short)"].any()}
        data_meta_lines.append(meta_line)

    dataset_meta = pd.DataFrame(data_meta_lines)
    return dataset_meta


def main():
    data_root = Path("/Users/raufkurbanov/Data/Aleksi")
    images_meta = extract_meta(data_root=data_root,
                               desc_path=data_root / "Aleksi.csv")
    dataset_meta = extract_dataset_split(images_meta)

    images_meta.to_csv(data_root / "meta.csv", index=None)
    dataset_meta.to_csv(data_root / "dataset.csv", index=None)


if __name__ == '__main__':
    main()
