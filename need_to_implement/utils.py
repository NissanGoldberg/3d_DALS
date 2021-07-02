import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

# TODO mask.detach() might cause problems
def euclidean_distance_transform(mask):
    epsilon = 0
    def bwdist(im): return distance_transform_edt(np.logical_not(im))
    bw = mask.detach().numpy()
    signed_dist = bwdist(bw) - bwdist(1 - bw)
    d = signed_dist
    d += epsilon
    while np.count_nonzero(d < 0) < 5:
        d -= 1

    return d


narrow_band_width = 1

def active_contour_layer(img, init_phi, map_lambda1_acl, map_lambda2_acl):
    wind_coef = 3
    zero_tensor = torch.zeros(1, dtype=torch.int)

    def _body(i, phi_level):
        band_index = torch.reduce_all([phi_level <= narrow_band_width, phi_level >= -narrow_band_width], axis=0)
        band = torch.where(band_index)
        band_y = band[:, 0]
        band_x = band[:, 1]
        shape_y = torch.shape(band_y)
        num_band_pixel = shape_y[0]
        window_radii_x = torch.ones(num_band_pixel) * wind_coef
        window_radii_y = torch.ones(num_band_pixel) * wind_coef


        def body_intensity(j, mean_intensities_outer, mean_intensities_inner):
            xnew = torch.cast(band_x[j], dtype="float32")
            ynew = torch.cast(band_y[j], dtype="float32")
            window_radius_x = torch.cast(window_radii_x[j], dtype="float32")
            window_radius_y = torch.cast(window_radii_y[j], dtype="float32")
            local_window_x_min = torch.cast(torch.floor(xnew - window_radius_x), dtype="int32")
            local_window_x_max = torch.cast(torch.floor(xnew + window_radius_x), dtype="int32")
            local_window_y_min = torch.cast(torch.floor(ynew - window_radius_y), dtype="int32")
            local_window_y_max = torch.cast(torch.floor(ynew + window_radius_y), dtype="int32")
            local_window_x_min = torch.maximum(zero_tensor, local_window_x_min)
            local_window_y_min = torch.maximum(zero_tensor, local_window_y_min)
            local_window_x_max = torch.minimum(torch.cast(input_image_size - 1, dtype="int32"), local_window_x_max)
            local_window_y_max = torch.minimum(torch.cast(input_image_size - 1, dtype="int32"), local_window_y_max)
            local_image = img[local_window_y_min: local_window_y_max + 1,local_window_x_min: local_window_x_max + 1]
            local_phi = phi_level[local_window_y_min: local_window_y_max + 1,local_window_x_min: local_window_x_max + 1]
            inner = torch.where(local_phi <= 0)
            area_inner = torch.cast(torch.shape(inner)[0], dtype='float32')
            outer = torch.where(local_phi > 0)
            area_outer = torch.cast(torch.shape(outer)[0], dtype='float32')
            image_loc_inner = torch.gather_nd(local_image, inner)
            image_loc_outer = torch.gather_nd(local_image, outer)
            mean_intensity_inner = torch.cast(torch.divide(torch.reduce_sum(image_loc_inner), area_inner), dtype='float32')
            mean_intensity_outer = torch.cast(torch.divide(torch.reduce_sum(image_loc_outer), area_outer), dtype='float32')
            mean_intensities_inner = torch.concat(axis=0, values=[mean_intensities_inner[:j], [mean_intensity_inner]])
            mean_intensities_outer = torch.concat(axis=0, values=[mean_intensities_outer[:j], [mean_intensity_outer]])

            return (j + 1, mean_intensities_outer, mean_intensities_inner)

    i = torch.constant(0, dtype=torch.int32)
    phi = init_phi
    _, phi = tf.while_loop(lambda i, phi: i < iter_limit, _body, loop_vars=[i, phi])
    phi = tf.round(tf.cast((1 - tf.nn.sigmoid(phi)), dtype=tf.float32))

    return phi, init_phi, map_lambda1_acl, map_lambda2_acl
