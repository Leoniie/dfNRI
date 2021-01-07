from . import encoders
from . import decoders
from . import nri
from . import dnri
from . import dnri_dynamicvars
import os
import torch
import torch.nn as nn


# https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html#
# https://github.com/pytorch/pytorch/issues/16885#issuecomment-551779897
class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def build_model(params):
    if params['model_type'] in ['dnri', 'dfnri']:
        dynamic_vars = params.get('dynamic_vars', False)
        if dynamic_vars:
            model = dnri_dynamicvars.DNRI_DynamicVars(params)
        else:
            model = dnri.DNRI(params)
        model_names = {'dnri':'dNRI', 'dfnri':'dfNRI'}
        print(f"{model_names[params['model_type']]} MODEL: ", model)

    else:
        num_vars = params['num_vars']
        graph_type = params['graph_type']

        # Build Encoder
        encoder = encoders.RefMLPEncoder(params)
        print("ENCODER: ",encoder)

        # Build Decoder
        decoder = decoders.GraphRNNDecoder(params)
        print("DECODER: ",decoder)
        if graph_type == 'dynamic':
            model = nri.DynamicNRI(num_vars, encoder, decoder, params)
        else:
            model = nri.StaticNRI(num_vars, encoder, decoder, params)

    if params['load_best_model']:
        print("LOADING BEST MODEL")
        path = os.path.join(params['working_dir'], 'best_model')
        model.load(path)
    elif params['load_model']:
        print("LOADING MODEL FROM SPECIFIED PATH")
        model.load(params['load_model'])
    if params['gpu']:
        # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html#create-model-and-dataparallel
        if torch.cuda.device_count() > 1 and not params['batch_size'] % torch.cuda.device_count():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = MyDataParallel(model)
        model.cuda()
    return model

