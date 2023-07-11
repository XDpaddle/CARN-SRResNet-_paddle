import paddle
import models.archs.SRResnet_arch as SRResNet_arch
import models.archs.classSR_rcan_arch as classSR_rcan_arch
import models.archs.RCAN_arch as RCAN_arch
import models.archs.CARN_arch as CARN_arch
# import models.archs.EDSR_arch as EDSR_arch

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'RCAN':
        netG = RCAN_arch.RCAN(n_resblocks=opt_net['n_resblocks'], n_feats=opt_net['n_feats'],
                              res_scale=opt_net['res_scale'], n_colors=opt_net['n_colors'],rgb_range=opt_net['rgb_range'],
                              scale=opt_net['scale'],reduction=opt_net['reduction'],n_resgroups=opt_net['n_resgroups'])
        
    elif which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    # elif which_model == 'RCAN':
    #     netG =EDSR_arch.EDSR(num_out_ch=opt_net['num_out_ch'], num_feat=opt_net['num_feat'],
    #                          num_block=opt_net['num_block'],upscale=opt_net['upscale'],res_scale=opt_net['res_scale'],
    #                           img_range=opt_net['img_range'],rgb_mean=opt_net['rgb_mean'])
    elif which_model == 'CARN_M':
            netG = CARN_arch.CARN_M(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                        nf=opt_net['nf'], scale=opt_net['scale'], group=opt_net['group'])
    elif which_model == 'classSR_3class_rcan':
        netG = classSR_rcan_arch.classSR_3class_rcan(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG