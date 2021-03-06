#from __future__ import division, absolute_import, print_function
import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
#import tensorflow as tf
#import keras.backend as K
#from keras.models import load_model

from detect.util import get_data, adv_dataset, get_model

from detect.attacks import (fast_gradient_sign_method, basic_iterative_method,
                            saliency_map_method)

# FGSM & BIM attack parameters that were chosen
ATTACK_PARAMS = {
    'mnist': {'eps': 0.300, 'eps_iter': 0.010},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010}
}

def craft_one_type(args, model, X, Y, attack):
    if attack == 'fgsm': # FGSM attack
        print('Crafting fgsm adversarial samples...')
        X_adv = fast_gradient_sign_method(
            model, X, Y,
            eps = ATTACK_PARAMS[args.dataset]['eps'],
            clip_min=0., clip_max=1.,
            batch_size = args.batch_size
        )

    elif attack in ['bim-a', 'bim-b']: # BIM attack
        print('Crafting %s adversarial samples...' % attack)
        its, results = basic_iterative_method(
            model, X, Y, eps = ATTACK_PARAMS[args.dataset]['eps'],
            eps_iter=ATTACK_PARAMS[args.dataset]['eps_iter'], clip_min=0.,
            clip_max=1., batch_size = args.batch_size
        )
        if attack == 'bim-a':
            # BIM-A, For each sample, select the time step where that sample first became misclassified
            X_adv = np.asarray( [results[its[i], i] for i in range(len(Y))] )
        else:
            # BIM-B, For each sample, select the very last time step
            X_adv = results[-1]


    elif attack == 'jsma':
        # JSMA attack
        print('Crafting jsma adversarial samples. This may take a while...')
        X_adv = saliency_map_method(
            model, X, Y, theta=1, gamma=0.1, clip_min=0., clip_max=1.
        )

    else:
        # TODO: CW attack
        raise NotImplementedError('CW attack not yet implemented.')

    print(X_adv)
    adv_data = adv_dataset(X_adv)
    #adv_loader = DataLoader(
    # 
    #)
    #acc = evaluate(model, 
    #        X_adv, Y, batch_size=batch_size, verbose=0)
    
    print("Model accuracy on the adversarial test set: %0.2f%%" % (100 * acc))
    np.save('../data/Adv_%s_%s.npy' % (args.dataset, args.attack), X_adv)



def evaluate(model, data_loader):
    model.eval()

    acc = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0]
            truth = batch[1]

            pred = model(x)
            batch_sz = x.shape[0]

            for i in range(batch_sz):
                if torch.argmax(pred[i], dim=0) == truth[i]:
                    acc += 1
                total += 1
    return acc / total




def main(args):
    assert args.dataset in ['mnist', 'cifar', 'svhn'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw', 'all'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma' or 'cw'"
    #assert os.path.isfile('../data/model_%s.h5' % args.dataset), \
    #    'model file not found... must first train model using train_model.py.'
    print('Dataset: %s. Attack: %s' % (args.dataset, args.attack))
    
    model = get_model(args.dataset)
    model.load_state_dict( torch.load(args.model) )
    
    train_dataset, test_dataset = get_data(args.dataset)
    test_loader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size
    )
    acc = evaluate(model, test_loader)
    print("Accuracy on the test set: %0.2f%%" % (100*acc))


    #X_test = [ torch.squeeze(tmp[0]).unsqueeze(-1) for tmp in test_dataset ]
    X_test = [ tmp[0] for tmp in test_dataset ]
    X_test = torch.stack(X_test, dim=0)
    print(X_test.shape)
    
    Y_test = [ torch.tensor([tmp[1]], dtype=torch.long) for tmp in test_dataset ]
    Y_test = torch.stack(Y_test, dim=0)
    print(Y_test.shape)

    if args.attack == 'all':
        # Cycle through all attacks
        for attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw']:
            craft_one_type(args, model, X_test, Y_test, attack)
    else:
        craft_one_type(args, model, X_test, Y_test, args.attack)

    print('Adversarial samples crafted and saved to data/ subfolder.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'jsma', 'cw' "
             "or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        default=128,
        required=False, type=int
    )
    parser.add_argument(
        '-m', '--model',
        help='load model path',
        type=str,
    )
    

    parser.set_defaults(batch_size=256)
    args = parser.parse_args()
    main(args)




