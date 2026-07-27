import geopandas
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging

from river_profile_lidar.water_speed_tools.create_water_speed import (
    process_and_save_water_spped,
)
from river_profile_lidar.river_name_mapping import get_river_save_name_from_index

log = logging.getLogger(__name__)

RASTER_RESOLUTION = 0.3


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def run(cfg: DictConfig):
    log.info(f"Doing Water Speed for: {cfg.river.name}")
    river_name = cfg.river.general_river_name
    rgb_image_folder_path = cfg.river.rgb_image_folder_path
    transect_path = cfg.river.transect_path
    index_file_path = cfg.river.index_file_path
    hab_folder_path = cfg.river.hab_output_path
    saving_folder_path = Path(cfg.river.water_speed_output_path)
    saving_folder_path.mkdir(parents=True, exist_ok=True)

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
        if Path(
            saving_folder_path, f"{Path(image_name).stem}_water_speed.tif"
        ).exists():
            continue
        if Path(hab_folder_path, f"{Path(image_name).stem}_hab.tif").exists() is False:
            continue
        rgb_image_path = Path(rgb_image_folder_path, image_name + ".tif")
        hab_image_path = Path(hab_folder_path, f"{Path(image_name).stem}_hab.tif")
        process_and_save_water_spped(
            rgb_image_path=rgb_image_path,
            transect_path=transect_path,
            hab_image_path=hab_image_path,
            raster_resolution=RASTER_RESOLUTION,
            output_path=saving_folder_path,
        )
        log.info(f"image: {image_name} Is Done")


if __name__ == "__main__":
    run()
