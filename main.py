from model import UNET
import torch
from DALS.pytorch.DALS.need_to_implement.utils import euclidean_distance_transform, active_contour_layer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--training', default=True, type=bool)
args = parser.parse_args()

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

if __name__ == '__main__':
    if args.training:
        x = torch.randn((1, 1, 256, 256))
        model = UNET()
        out_seg = model(x)
        # print(out_seg.shape)
        # print(x.shape)
        # print(model)
        # image = torch.rand((1, 572, 572))
        # summary(model, (1, 256, 256))
        # assert out_seg.shape == x.shape

        map_lambda1 = torch.exp((2.0 - out_seg) / (1.0 + out_seg))
        map_lambda2 = torch.exp((1.0 + out_seg) / (2.0 - out_seg))
    else:  # not training, then restore
        pass

    y_out_dl = torch.round(out_seg)
    x_acm = x[:, 0, :, :]
    rounded_seg_acl = y_out_dl[:, 0, :, :]  # get rid of channels
    print("rounded_seg_acl ", rounded_seg_acl.shape)
    # shape is 1, 256, 256
    # should be rounded_seg_acl (1, 512, 512)

    dt_trans = euclidean_distance_transform(rounded_seg_acl)

    # map each x_acm batch to function
    # phi_out, _, lambda1_tr, lambda2_tr = active_contour_layer(img=x_acm,
    #                                                           init_phi=dt_trans,
    #                                                           map_lambda1_acl=map_lambda1[:, 0, :, :],
    #                                                           map_lambda2_acl=map_lambda2[:, 0, :, :])


    # phi_out, _, lambda1_tr, lambda2_tr = torch.vmap(model)(func=active_contour_layer)
    # x = range(x_acm.shape[0])
    # [active_contour_layer(n) for n in x]

    active_contour_layers = []
    # iterate through each batch

    # for i in range(x_acm.shape[0]):
    #     phi_out, _, lambda1_tr, lambda2_tr = active_contour_layer(img=x_acm[i, :, :, :],
    #                                                               init_phi=dt_trans,
    #                                                               map_lambda1_acl=map_lambda1[:, 0, :, :],
    #                                                               map_lambda2_acl=map_lambda2[:, 0, :, :])

    # img = x_acm[0, :, :, :] is first batch
    active_contour_layer(img=x_acm[0, :, :, :],
                         init_phi=dt_trans,
                         map_lambda1_acl=map_lambda1[:, 0, :, :],
                         map_lambda2_acl=map_lambda2[:, 0, :, :])

