from pathlib import Path

import pandas as pd
import argparse


def extract_meta(data_root: Path, meta_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(meta_path)

    new_meta_rows = []

    for tif_path in data_root.glob("**/*.tif"):
        tif_name = tif_path.name

        is_thumb = tif_name.find("thumb") != -1
        if is_thumb:
            continue

        plate_barcode = tif_name.split("_")[0].split("-")[0]
        well_id = tif_name.split("_")[1]
        channel = int(tif_name.split("_")[-1][1])
        relative_path = str(tif_path.relative_to(data_root))
        sample = tif_name.split("_")[2]

        new_meta_row = {"plate_barcode": plate_barcode,
                        "well_id": well_id,
                        "channel_path": relative_path,
                        "sample_id": sample,
                        "channel": channel}
        new_meta_rows.append(new_meta_row)
    new_meta = pd.DataFrame(new_meta_rows)

    new_meta_ch1 = new_meta[new_meta.channel == 1]
    new_meta_ch2 = new_meta[new_meta.channel == 2]
    merged_meta = new_meta_ch1.merge(new_meta_ch2, on=["plate_barcode", "well_id", "sample_id"])
    merged_meta = merged_meta.drop(["channel_x", "channel_y", "sample_id"], axis=1)

    old_meta_filtered = meta[["plate_barcode", "well_id", "cmpd", "conc_uM", "treatment_id", "cmpd_name"]]
    final_meta = merged_meta.merge(old_meta_filtered, how="left", on=["plate_barcode", "well_id"])

    return final_meta


def flag_damaged(meta: pd.DataFrame, flagged_df: pd.DataFrame) -> pd.DataFrame:
    flagged_samples = meta.channel_path_x.apply(lambda x: x.split("/")[-1]).isin(flagged_df.FileName_HOECHST)
    masked_meta = meta.assign(damaged=flagged_samples)

    return masked_meta


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Script to process the metadata from biologists "
                                                 "to a more convinient format")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--meta_path", type=Path, required=True)
    parser.add_argument("--flagged_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)

    return parser


def main():
    # data_root = Path("/Users/raufkurbanov/Data/Ola")
    # meta_path = Path("/Users/raufkurbanov/Data/Ola/metadata/PVE-plateLayouts.csv")
    # flagged_path = Path("/Users/raufkurbanov/Data/Ola/List_Flagged_Images.csv")
    # save_path = Path("/Users/raufkurbanov/Data/Ola/metadata/dataset.csv")
    parser = get_parser()
    params = parser.parse_args()
    data_root: Path = params.data_root
    meta_path: Path = params.meta_path
    flagged_path: Path = params.flagged_path
    save_path: Path = params.save_path

    dataset_meta = extract_meta(data_root, meta_path)
    flagged_meta = pd.read_csv(flagged_path)
    flagged_dataset_meta = flag_damaged(dataset_meta, flagged_meta)
    flagged_dataset_meta.to_csv(save_path, index=None)


if __name__ == '__main__':
    main()
