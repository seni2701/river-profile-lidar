import geopandas
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging

from river_profile_lidar.bathymetry.trapezoid.create_trapezoid import (
    process_and_save_trapezoid_bathymetry,
)

log = logging.getLogger(__name__)

RASTER_RESOLUTION = 0.3


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def run(cfg: DictConfig):
    log.info(f"Doing Trapezoid for: {cfg.river.name}")

    rgb_image_folder_path = Path(cfg.river.rgb_image_folder_path)
    transect_path = Path(cfg.river.transect_path)
    index_file_path = Path(cfg.river.index_file_path)
    cross_sections_points_path = Path(
        cfg.river.cross_section_output_path, "trapezoid_cross_section_points.shp"
    )
    saving_folder_path = Path(cfg.river.trapezoid_output_path)

    image_to_process_df = geopandas.read_file(index_file_path)

    hierarchie_col_name = (
        "Hierarchie" if "Hierarchie" in image_to_process_df.columns else "HIERARCHIE"
    )
    image_to_process_df = image_to_process_df.sort_values(
        by=hierarchie_col_name
    ).reset_index(drop=True)

    # 🔴 CORRECTION UNIQUE ICI :
    # on prend le nom EXACT de l’image (sans 4bandes) + .tif
    image_to_process_df["nom_image_save"] = image_to_process_df["NOM_IMAGE"]

    image_name_to_process_list = image_to_process_df["nom_image_save"].to_list()
    number_image_to_process = len(image_name_to_process_list)

    for i, image_name in enumerate(image_name_to_process_list):
        log.info(f"image: {image_name} {i+1}/{number_image_to_process}")

        rgb_image_path = Path(rgb_image_folder_path, image_name + ".tif")

        if not rgb_image_path.exists():
            log.warning(f"Image introuvable : {rgb_image_path}")
            continue

        saving_folder_path.mkdir(parents=True, exist_ok=True)
        output_raster = Path(
            saving_folder_path, f"{rgb_image_path.stem}_trapezoid.tif"
        )

        if output_raster.exists():
            log.info(f"Déjà traité : {output_raster.name}")
            continue

        process_and_save_trapezoid_bathymetry(
            rgb_image_path=rgb_image_path,
            transect_path=transect_path,
            cross_sections_points_path=cross_sections_points_path,
            raster_resolution=RASTER_RESOLUTION,
            output_path=saving_folder_path,
        )

        log.info(f"image: {image_name} DONE")


if __name__ == "__main__":
    run()
