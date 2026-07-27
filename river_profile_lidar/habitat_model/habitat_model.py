import numpy


class HabitatModel:
    def __init__(
        self,
        global_hist_envelop_curve_2d,
        global_hist_envelop_curve_2d_edges,
        global_hist_envelop_curve_1d,
        global_hist_envelop_curve_1d_edges,
    ):
        global_hist_envelop_curve_2d = numpy.pad(
            global_hist_envelop_curve_2d,
            ((0, 1), (0, 1)),
            mode="constant",
            constant_values=0,
        )
        self.global_hist_envelop_curve_2d = global_hist_envelop_curve_2d
        self.global_hist_envelop_curve_2d_edges = global_hist_envelop_curve_2d_edges
        global_hist_envelop_curve_1d = numpy.append(global_hist_envelop_curve_1d, 0)
        self.global_hist_envelop_curve_1d = global_hist_envelop_curve_1d
        self.global_hist_envelop_curve_1d_edges = global_hist_envelop_curve_1d_edges

    def get_hist_1d_estimation_from_value(self, d84):
        bin = numpy.digitize(d84, self.global_hist_envelop_curve_1d_edges) - 1
        hist_estimation = self.global_hist_envelop_curve_1d[bin]
        return hist_estimation

    def get_hist_2d_estimation_from_value(self, profondeur, vitesse):
        x_bin = (
            numpy.digitize(profondeur, self.global_hist_envelop_curve_2d_edges[0]) - 1
        )
        y_bin = numpy.digitize(vitesse, self.global_hist_envelop_curve_2d_edges[1]) - 1
        hist_estimation = self.global_hist_envelop_curve_2d[x_bin, y_bin]
        return hist_estimation
