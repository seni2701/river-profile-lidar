import geopandas
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging

from river_profile_lidar.d84_tools.create_d84 import process_and_save_d84

log = logging.getLogger(__name__)

RASTER_RESOLUTION = 0.3


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def run(cfg: DictConfig):
    log.info(f"Doing D84 for: {cfg.river.name}")

    rgb_image_folder_path = cfg.river.rgb_image_folder_path
    transect_path = cfg.river.transect_path
    index_file_path = cfg.river.index_file_path
    hab_folder_path = cfg.river.hab_output_path
    saving_folder_path = Path(cfg.river.d84_output_path)

    image_to_process_df = geopandas.read_file(index_file_path)

    hierarchie_col_name = (
        "Hierarchie" if "Hierarchie" in image_to_process_df.columns else "HIERARCHIE"
    )

    image_to_process_df = image_to_process_df.sort_values(
        by=hierarchie_col_name
    ).reset_index(drop=True)

    # ✅ CORRECTION CLÉ : conserver le nom réel des images
    image_to_process_df["nom_image_save"] = (
        image_to_process_df["NOM_IMAGE"]
    )

    image_name_to_process_list = image_to_process_df["nom_image_save"].to_list()
    number_image_to_process = len(image_name_to_process_list)

    for i, image_name in enumerate(image_name_to_process_list):
        log.info(f"image: {image_name} {i+1}/{number_image_to_process}")

        image_name_stem = Path(image_name).stem

        # sortie D84 déjà existante
        if Path(saving_folder_path, f"{image_name_stem}_d84.tif").exists():
            continue

        # dépendance HAB obligatoire
        if not Path(hab_folder_path, f"{image_name_stem}_hab.tif").exists():
            log.warning(f"HAB manquant pour : {image_name_stem}")
            continue

        rgb_image_path = Path(rgb_image_folder_path, image_name + ".tif")
        saving_folder_path.mkdir(parents=True, exist_ok=True)

        process_and_save_d84(
            rgb_image_path=rgb_image_path,
            transect_path=transect_path,
            raster_resolution=RASTER_RESOLUTION,
            output_path=saving_folder_path,
        )

        log.info(f"image: {image_name} Is Done")


if __name__ == "__main__":
    run()
