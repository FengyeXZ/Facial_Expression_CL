# Based on https://github.com/aimagelab/mammoth
import os
import importlib


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('model')
            if not model.find('__') > -1 and 'py' in model]


names = {}
for model in get_all_models():
    # print(get_all_models())
    mod = importlib.import_module('model.' + model)
    class_name = {x.lower(): x for x in mod.__dir__()}[model.replace('_', '')]
    # print(class_name)
    names[model] = getattr(mod, class_name)


def get_model(args, backbone, loss, transform):
    return names[args.model](backbone, loss, args, transform)
