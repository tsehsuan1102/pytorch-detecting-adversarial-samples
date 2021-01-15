import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchattacks
import pickle

from detect.util import get_data, get_model, evaluate
from detect.attacks import (fast_gradient_sign_method,
                            basic_iterative_method,
                            saliency_map_method)


# FGSM & BIM attack parameters that were chosen
ATTACK_PARAMS = {
    'mnist': {'eps': 1, 'eps_iter': 0.010},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010}
}



def craft_one_type(args, model, X, Y, attack, test_loader=None):
    Y = Y.squeeze()
    
    if attack == 'fgsm': # FGSM attack
        print('Crafting fgsm adversarial samples...')
        eps     = ATTACK_PARAMS[args.dataset]['eps']
        #eps = 0.03
        X_adv   = fast_gradient_sign_method(model, X, Y, eps, test_loader)

    elif attack in ['bim-a', 'bim-b', 'bim']: # BIM attack
        print('Crafting %s adversarial samples...' % attack)
        eps         = ATTACK_PARAMS[args.dataset]['eps']
        eps_iter    = ATTACK_PARAMS[args.dataset]['eps_iter']
        X_adv       = basic_iterative_method(model, X, Y, eps, eps_iter, test_loader)

    #elif attack == 'jsma':
    #    # JSMA attack
    #    print('Crafting jsma adversarial samples. This may take a while...')
    #    X_adv = saliency_map_method(
    #        model, X, Y, theta=1, gamma=0.1, clip_min=0., clip_max=1.
    #    )
    #else:
    #    #TODO: CW attack
    #    raise NotImplementedError('幹您娘CW attack not yet implemented.')

    print(X_adv.shape)
    adv_data = [(x_now, y_now) for x_now, y_now in zip(X_adv, Y) ]
    adv_loader = DataLoader(
        dataset = adv_data,
        batch_size = args.batch_size,
    )
    acc = evaluate(model, adv_loader)
    print("Model accuracy on the adversarial test set: %0.2f%%" % (100 * acc))
    np.save('../data/Adv_%s_%s_train.npy' % (args.dataset, args.attack), X_adv)



def main(args):
    #### assertions
    assert args.dataset in ['mnist', 'cifar', 'svhn'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'bim', 'jsma', 'cw', 'all'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma' or 'cw'"
    assert os.path.isfile(args.model), \
        'model file not found... must first train model using train_model.py.'
    print('Dataset: %s. Attack: %s' % (args.dataset, args.attack))
    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    ## load model
    model = get_model(args.dataset)
    model.load_state_dict( torch.load(args.model) )
    model.to(device)

    #train_dataset = get_data(args.dataset, train=True)
    test_dataset = get_data(args.dataset, train=True)
    
    ## evaluate on normal testset
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = args.batch_size
    )
    acc = evaluate(model, test_loader)
    print("Accuracy on the test set: %0.2f%%" % (100*acc))


    X_test = [ tmp[0] for tmp in test_dataset ]
    X_test = torch.stack(X_test, dim=0)
    print('X: ', X_test.shape)
    
    Y_test = [ torch.tensor([tmp[1]], dtype=torch.long) for tmp in test_dataset ]
    Y_test = torch.stack(Y_test, dim=0)
    print('Y:', Y_test.shape)

    ## craft 
    craft_one_type(args, model, X_test, Y_test, args.attack, test_loader)
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
        required=True,
        type=str,
    )
    parser.set_defaults(batch_size=128)
    args = parser.parse_args()
    main(args)




