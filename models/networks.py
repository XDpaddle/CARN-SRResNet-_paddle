import paddle
import models.archs.SRResnet_arch as SRResNet_arch
# import models.archs.classSR_rcan_arch as classSR_rcan_arch
# import models.archs.RCAN_arch as RCAN_arch
import models.archs.CARN_arch as CARN_arch


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
        
    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])

    elif which_model == 'CARN_M':
            netG = CARN_arch.CARN_M(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                        nf=opt_net['nf'], scale=opt_net['scale'], group=opt_net['group'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG