def get_river_save_name_from_index(river_name: str):
    """
    In the Index file, the filename doesn't always match the the saved image name
    This function is to retrieve the saved name based on the filename in the index file.
    Each river needs to have is own definition
    """

    if river_name == "ESC":
        return get_river_save_name_from_index_esc
    if river_name == "MAN":
        return get_river_save_name_from_index_man
    if river_name == "DMT":
        return get_river_save_name_from_index_dmt
    if river_name == "YRK":
        return get_river_save_name_from_index_yrk
    if river_name == "STJ" or river_name == "STJS":
        return get_river_save_name_from_index_stj
    if river_name == "TRI":
        return get_river_save_name_from_index_tri
    if river_name == "MAP":
        return get_river_save_name_from_index_map
    if river_name == "SMRNE":
        return get_river_save_name_from_index_smrne
    if river_name == "SMR":
        return get_river_save_name_from_index_smr
    if river_name == "FER":
        return get_river_save_name_from_index_fer
    else:
        raise ValueError("River not Implemented")


def get_river_save_name_from_index_dmt(nom_image):
    save_name = f"{nom_image.replace('_rgb', '')}_4bandes_ortho.tif"
    return save_name


def get_river_save_name_from_index_man(nom_image):
    save_name = f"{nom_image.replace('_rgb', '')}_4bandes_ortho.tif"
    return save_name


def get_river_save_name_from_index_esc(nom_image):
    save_name = nom_image
    return save_name


def get_river_save_name_from_index_yrk(nom_image):
    save_name = f"{nom_image.replace('_rgb', '').replace('_nir', '')}_4bandes_ortho.tif"
    return save_name


def get_river_save_name_from_index_stj(nom_image):
    save_name = f"{nom_image.replace('_rgb', '')}_4bandes_ortho.tif"
    return save_name


def get_river_save_name_from_index_tri(nom_image):
    save_name = f"{nom_image[:-4].upper()}.tif"
    return save_name


def get_river_save_name_from_index_map(nom_image):
    save_name = nom_image
    return save_name


def get_river_save_name_from_index_smrne(nom_image):
    save_name = "_".join(nom_image.split("_")[:2])
    save_name = f"{save_name}_4bandes_ortho.tif"
    save_name = save_name.capitalize()
    return save_name


def get_river_save_name_from_index_smr(nom_image):
    if "rgb" in nom_image:
        save_name = f"{nom_image.replace('_rgb', '')}_4bandes_ortho.tif".capitalize()
    else:
        save_name = nom_image.replace(" ", "")
    return save_name
