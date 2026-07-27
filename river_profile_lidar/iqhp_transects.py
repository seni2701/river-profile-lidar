import geopandas
import pickle
from pathlib import Path
import rasterio
from tqdm import tqdm
from shapely.geometry import Polygon
from rasterstats import zonal_stats
import logging
import hydra
from omegaconf import DictConfig
from river_profile_lidar.bathymetry.hab.utils import calculate_da
from habitat_model import HabitatModel  # noqa: F401


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def run(cfg: DictConfig):
    log.info(f"Doing IQH Transects for: {cfg.river.name}")
    iqh_image_folder_path = Path(cfg.river.iqh_output_path)
    transect_path = Path(cfg.river.transect_path)
    with open("habitat_model.pkl", "rb") as f:
        habitat_model = pickle.load(f)

    all_iqh_image_path = list(
        filter(
            lambda path: path.suffix == ".tif", Path(iqh_image_folder_path).iterdir()
        )
    )
    saving_path = Path(cfg.river.transect_iqh_output_path)
    transect_polygon_df = geopandas.read_file(transect_path)
    transect_polygon_df = transect_polygon_df[transect_polygon_df["Backwater"] == 0]
    transect_polygon_df = transect_polygon_df[transect_polygon_df["Lac"] == 0]
    transect_polygon_df = transect_polygon_df[~transect_polygon_df["Q_IMG_spli"].isna()]
    transect_polygon_df = transect_polygon_df.sort_values(by="PK").reset_index(
        drop=True
    )
    surface_col_name = (
        "surface" if "surface" in transect_polygon_df.columns else "Surface"
    )
    transect_polygon_df["mean_iqhp"] = None
    transect_polygon_df["UPHP"] = None
    for iqh_image_path in tqdm(all_iqh_image_path):
        with rasterio.open(iqh_image_path, "r") as src:
            iqh_array = src.read(1)
            transform = src.transform
            iqh_bounds = src.bounds

        iqh_polygon = Polygon(
            [
                (iqh_bounds.left, iqh_bounds.top),
                (iqh_bounds.left, iqh_bounds.bottom),
                (iqh_bounds.right, iqh_bounds.bottom),
                (iqh_bounds.right, iqh_bounds.top),
                (iqh_bounds.left, iqh_bounds.top),
            ]
        )
        for i, row in transect_polygon_df.iterrows():
            if row["geometry"].within(iqh_polygon) is True:
                transect_stats = zonal_stats(
                    row["geometry"],
                    iqh_array,
                    affine=transform,
                    nodata=None,
                    stats=["count", "mean"],
                )[0]
                if transect_stats["count"] > 0:
                    transect_polygon_df.loc[i, "mean_iqhp"] = transect_stats["mean"]

    min_non_zero_slope = transect_polygon_df[transect_polygon_df["Pente"] > 0][
        "Pente"
    ].min()
    zero_or_nan_slope_idx = (transect_polygon_df["Pente"].isna()) | (
        transect_polygon_df["Pente"] <= 0
    )
    transect_polygon_df.loc[zero_or_nan_slope_idx, "Pente"] = min_non_zero_slope

    transect_polygon_df["DA"] = transect_polygon_df.apply(
        lambda row: calculate_da(row["Q_IMG_spli"], row["Pente"], row["WAT_WIDTH"]),
        axis=1,
    )
    transect_polygon_df["Vi"] = (
        transect_polygon_df["Q_IMG_spli"] / transect_polygon_df["WAT_WIDTH"]
    )

    transect_polygon_df["iqhp_1d_transect"] = transect_polygon_df["D84"].apply(
        habitat_model.get_hist_1d_estimation_from_value
    )
    transect_polygon_df["iqhp_2d_transect"] = transect_polygon_df.apply(
        lambda row: habitat_model.get_hist_2d_estimation_from_value(
            row["DA"], row["Vi"]
        ),
        axis=1,
    )
    transect_polygon_df["iqhp_transect"] = (
        transect_polygon_df["iqhp_1d_transect"]
        * transect_polygon_df["iqhp_2d_transect"]
    )

    nan_mean_iqh_idx = transect_polygon_df["mean_iqhp"].isna()
    transect_polygon_df.loc[nan_mean_iqh_idx, "mean_iqhp"] = transect_polygon_df.loc[
        nan_mean_iqh_idx, "iqhp_transect"
    ]
    transect_polygon_df["UPHP"] = (
        transect_polygon_df["mean_iqhp"] * transect_polygon_df[surface_col_name]
    )
    transect_polygon_df["UPHP"] = transect_polygon_df["UPHP"].astype(float)
    transect_polygon_df = transect_polygon_df.drop(
        columns=["iqhp_1d_transect", "iqhp_2d_transect", "iqhp_transect"]
    )
    schema = geopandas.io.file.infer_schema(transect_polygon_df)
    schema["properties"]["FACC_M2"] = "float:20.5"
    schema["properties"]["FACC_COR"] = "float:20.5"
    schema["properties"]["FACC_CORR"] = "float:20.5"

    saving_path.parent.mkdir(parents=True, exist_ok=True)
    transect_polygon_df.to_file(saving_path)


if __name__ == "__main__":
    run()
