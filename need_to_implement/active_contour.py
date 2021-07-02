import torch
iter_limit = 300

#  args
#  1st: original image 256^2 /maybe output of UNet,
#  2nd: distance map - initial contour: distance map
#  3rd, 4th: lambda
def active_contour_layer(img, init_phi, map_lambda1_acl, map_lambda2_acl):
    wind_coef = 3
    zero_tensor = torch.tensor(0, dtype=torch.int32)

    i = 0
    phi = init_phi

    def _body(i, phi_level):
        band_index = tf.reduce_all([phi_level <= narrow_band_width, phi_level >= -narrow_band_width],
                                   axis=0)  # "logical and" of elements across dimensions
        pass

    # when does this stop??? needs i
    while i < iter_limit:
        _body(i, phi)


