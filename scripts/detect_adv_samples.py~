import os
import argparse
import warnings
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import KernelDensity

from detect.util import (get_data, get_model, get_noisy_samples, get_mc_predictions,
                         get_deep_representations, score_samples, normalize,
                         train_lr, compute_roc,
                         get_value,
                         AddGaussianNoise, evaluate)

# Optimal KDE bandwidths that were determined from CV tuning
BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26, 'svhn': 1.00}


def getXY(dataset):
    X = []
    Y = []
    for x, y in dataset:
        X.append(x)
        Y.append(torch.tensor(y))
    #print(X, Y)

    X = torch.stack(X)
    Y = torch.stack(Y)
    return np.array(X), np.array(Y)


def evaluate_test(args, model, kdes, datatypes, nb_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = {}
    
    transform = None
    test_dataset['normal'] = get_data(args.dataset, train=False, transform=transform)
    test_dataset['noisy'] = get_data(args.dataset, train=False, transform=transform)
    ##### Load adversarial samples (create by crate_adv_samples.py)
    ### craft adversarial examples on test_dataset
    print('[Test] Loading noisy and adversarial samples...')
    X_test_adv = np.load('../data/Adv_%s_%s.npy' % (args.dataset, args.attack))
    X_test_adv = torch.from_numpy(X_test_adv)
    test_adv   = [ (x_tmp, y_tmp[1]) for x_tmp, y_tmp in zip(X_test_adv, test_dataset['normal']) ]
    test_dataset['adversarial'] = test_adv


    num = len(test_dataset['normal'])
    
    test_loader = {}
    for datatype in datatypes:
        test_loader[datatype] = DataLoader(
            dataset = test_dataset[datatype],
            batch_size = args.batch_size,
            shuffle = False
        )
    ### TODO pick data(model predict correctly on normal)
    ### get uncertainty
    print('[Test] Getting Monte Carlo dropout variance predictions...')
    uncerts = {}
    for datatype in datatypes:
        uncerts[datatype] = get_mc_predictions(model, test_loader[datatype], nb_iter=nb_size, method=args.uncert)

    ################# Get KDE scores
    # Get deep feature representations
    print('[Test] Getting deep feature representations...')
    features = {}
    for datatype in datatypes:
        features[datatype] = get_deep_representations(model, test_loader[datatype], args.dataset)

    # Get model predictions
    print('[Test] Computing model predictions...')
    preds = {}
    for datatype in datatypes:
        with torch.no_grad():
            tmp_result = []
            for batch in test_loader[datatype]:
                x = batch[0].to(device)
                pred = model(x)
                tmp_result.append(pred.detach().cpu())
            preds[datatype] = torch.argmax( torch.cat(tmp_result), dim=1 )

    # Get density estimates
    ###### get test density
    print('[Test] computing densities...')
    densities = {}
    for datatype in datatypes:
        densities[datatype] = score_samples(
            kdes,
            features[datatype].cpu(),
            preds[datatype].cpu()
        )
    ###### Z-score the uncertainty and density values
    ###### normalize
    uncerts_z = {}
    uncerts_z['normal'], uncerts_z['noisy'], uncerts_z['adversarial'] = normalize(
        uncerts['normal'].cpu().numpy(),
        uncerts['noisy'].cpu().numpy(),
        uncerts['adversarial'].cpu().numpy(),
    )
    densities_z = {}
    densities_z['normal'], densities_z['noisy'], densities_z['adversarial'] = normalize(
        densities['normal'],
        densities['noisy'],
        densities['adversarial'],
    )
    print('.......Densities............')
    for datatype in datatypes:
        print(datatype, ' Mean: ', densities_z[datatype].mean() )

    ### dense, uncert, combine
    flags = ['dense', 'uncert', 'combine']
    values  = {}
    labels  = {}
    for flag in flags:
        tmp_values, tmp_labels = get_value(
            densities_pos   = densities_z['adversarial'],
            densities_neg   = np.concatenate((densities_z['normal'], densities_z['noisy'])),
            uncerts_pos     = uncerts_z['adversarial'],
            uncerts_neg     = np.concatenate((uncerts_z['normal'], uncerts_z['noisy'])),
            #entro_pos       = entro_z['adversarial'],
            #entro_neg       = np.concatenate((entro_z['normal'], entro_z['noisy'])),
            flag = flag
        )
        values[flag] = tmp_values
        labels[flag] = tmp_labels

    return values, labels, num



def main(args):
    datatypes   = ['normal', 'noisy', 'adversarial']
    ## assertions
    assert args.dataset in ['mnist', 'cifar', 'svhn'], "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'bim', 'jsma', 'cw', 'all'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma' or 'cw'"
    #assert os.path.isfile('../data/Adv_%s_%s.npy' % (args.dataset, args.attack)), \
    #    'adversarial sample file not found... must first craft adversarial ' \
    #    'samples using craft_adv_samples.py'

    print('Loading the data and model...')
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = get_model(args.dataset)
    model.load_state_dict(torch.load(args.model))
    model.to(device)

    model.eval()
    # Load the dataset
    train_data  = get_data(args.dataset, train=True)
    print(train_data[0])
    train_loader = DataLoader(
        dataset = train_data,
        batch_size = args.batch_size,
        shuffle = False
    )

    ##### Load adversarial samples (create by crate_adv_samples.py)
    print('Loading noisy and adversarial samples...')
    
    ### train_adv
    X_train_adv = np.load('../data/Adv_%s_%s_train.npy' % (args.dataset, args.attack))
    X_train_adv = torch.from_numpy(X_train_adv)
    train_adv   = [ (x_tmp, y_tmp[1]) for x_tmp, y_tmp in zip(X_train_adv, train_data) ]
    
    ##### create noisy data
    if args.dataset == 'mnist':
        noise_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.1307, ), std = (0.3081, )),
            AddGaussianNoise(0., 0.1)
        ])
    elif args.dataset == 'cifar':
        noise_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.4914, 4822, 4465), std = (0.247, 0.243, 0.261) ),
            #AddGaussianNoise(0., 0.1)
        ])
    train_noisy = get_data(args.dataset, train=True)#, transform=noise_transform)
    print('NOISY', train_noisy)
    print(train_noisy[0])

    
    X_train, Y_train = getXY(train_data)
    # Check model accuracies on each sample type
    for s_type, dataset in zip(['normal',   'noisy',    'adversarial'],
                               [train_data, train_noisy, train_adv]):
        data_loader = DataLoader(
            dataset = dataset,
            batch_size = args.batch_size,
        )
        acc = evaluate(model, data_loader)
        print("Model accuracy on the %s test set: %0.2f%%" % (s_type, 100 * acc) )
        # Compute and display average perturbation sizes
        ### TODO
        X_now, Y_now = getXY(dataset)

        if not s_type == 'normal':
            #print( X_now.reshape((len(X_train), -1)) - X_train.reshape((len(X_train), -1)) )
            #print( X_now.reshape((len(X_train), -1)).max() )
            #print( X_now.reshape((len(X_train), -1)).mean() )
            #print( X_now.reshape((len(X_train), -1)).min() )
            #print( X_train.reshape((len(X_train), -1)).max() )
            #print( X_train.reshape((len(X_train), -1)).mean() )
            #print( X_train.reshape((len(X_train), -1)).min() )
            l2_diff = np.linalg.norm(
                X_now.reshape((len(X_train), -1)) - X_train.reshape((len(X_train), -1)),
                axis=1
            ).mean()
            print("Average L-2 perturbation size of the %s test set: %0.2f" % (s_type, l2_diff))

    ### Refine the normal, noisy and adversarial sets to only include samples for which the original version was correctly classified by the model
    ### run test data and choose the data model can correctly predict
    y_train_list    = []
    pred_train_list = []
    with torch.no_grad():
        for batch in train_loader:
            x       = batch[0].to(device)
            y_train_list.append(batch[1])
            pred_train_list.append( model(x) )

    y_train_list = torch.cat(y_train_list)
    Y_train = torch.tensor(y_train_list).detach().cpu()
    pred_train = torch.cat(pred_train_list).detach().cpu()

    inds_correct = torch.where(Y_train == pred_train.argmax(axis=1), torch.full_like(Y_train, 1), torch.full_like(Y_train, 0)).to(device)

    
    picked_train_data = {}
    for datatype in datatypes:
        picked_train_data[datatype] = []
    for i, (b, y_tmp) in enumerate(zip(inds_correct, Y_train)):
        if b == 1:
            picked_train_data['normal'].append( (train_data[i][0], y_tmp) )
            picked_train_data['noisy'].append( (train_noisy[i][0], y_tmp) )
            picked_train_data['adversarial'].append( (X_train_adv[i], y_tmp) )
        else:
            continue

    picked_train_loader = {}
    for datatype in datatypes:
        picked_train_loader[datatype] = DataLoader(
            dataset = picked_train_data[datatype],
            batch_size = args.batch_size
        )


    ###########################################################################################################################################
    ################# Get Bayesian uncertainty scores
    nb_size = 50
    print('Getting Monte Carlo dropout variance predictions...')
    uncerts = {}
    test_uncerts = {}
    for datatype in datatypes:
        uncerts[datatype] = get_mc_predictions(model, picked_train_loader[datatype], nb_iter=nb_size, method=args.uncert)

    ################# Get KDE scores
    # Get deep feature representations
    print('Getting deep feature representations...')
    x_train_features         = get_deep_representations(model, train_loader              , args.dataset)
    picked_train_features = {}
    for datatype in datatypes:
        picked_train_features[datatype] = get_deep_representations(model, picked_train_loader[datatype], args.dataset)
    
    print('Shape')
    print(x_train_features.shape)
    for datatype in datatypes:
        print(picked_train_features[datatype].shape)
    ####### CLASS NUM ########
    class_num = 10
    Y_train_label   = [ tmp[1] for tmp in train_data ]
    Y_train_label   = np.array(Y_train_label)
    Y_train         = np.zeros((len(Y_train_label), class_num))
    Y_train[ np.arange(Y_train_label.size), Y_train_label ] = 1
    # Train one KDE per class
    print('Training KDEs...')
    class_inds = {}
    for i in range(class_num):
        class_inds[i] = np.where(Y_train_label == i)[0]
        print('class_inds[', i, ']: ', class_inds[i].size )
    kdes = {}
    warnings.warn("Using pre-set kernel bandwidths that were determined optimal for the specific CNN models of the paper. If you've "
                  "changed your model, you'll need to re-optimize the bandwidth.")
    
    ### Use train features to fit Kernel density
    for i in range(class_num):
        kdes[i] = KernelDensity(kernel='gaussian', bandwidth=BANDWIDTHS[args.dataset]).fit( x_train_features.cpu().numpy()[class_inds[i]] )

    # Get model predictions
    print('Computing model predictions...')
    data_loaders = []
    for datatype in datatypes:
        data_loaders.append(picked_train_loader[datatype])
    preds = []
    preds_train = {}
    for now_loader in data_loaders:
        with torch.no_grad():
            tmp_result = []
            for batch in now_loader:
                x = batch[0].to(device)
                pred = model(x)
                tmp_result.append(pred.detach().cpu())
            preds.append( torch.cat(tmp_result) )
    preds_train['normal']       = torch.argmax(preds[0], dim=1)
    preds_train['noisy']        = torch.argmax(preds[1], dim=1)
    preds_train['adversarial']  = torch.argmax(preds[2], dim=1)

    # Get density estimates
    ###### get test density
    print('computing densities...')
    train_densities = {}
    for datatype in datatypes:
        train_densities[datatype] = score_samples(
            kdes,
            picked_train_features[datatype].cpu(),
            preds_train[datatype].cpu()
        )
    ###### Z-score the uncertainty and density values
    ###### normalize
    uncerts_z = {}
    uncerts_z['normal'], uncerts_z['noisy'], uncerts_z['adversarial'] = normalize(
        uncerts['normal'].cpu().numpy(),
        uncerts['noisy'].cpu().numpy(),
        uncerts['adversarial'].cpu().numpy(),
    )
    densities_z = {}
    densities_z['normal'], densities_z['noisy'], densities_z['adversarial'] = normalize(
        train_densities['normal'],
        train_densities['noisy'],
        train_densities['adversarial'],
    )
    print('.......Densities............')
    for datatype in datatypes:
        print(datatype, ' Mean: ', densities_z[datatype].mean() )
    
    ## Build detector
    ### dense, uncert, combine
    flags = ['dense', 'uncert', 'combine']
    values  = {}
    labels  = {}
    lrs     = {}
    for flag in flags:
        tmp_values, tmp_labels, tmp_lr = train_lr(
            densities_pos = densities_z['adversarial'],
            densities_neg = np.concatenate((densities_z['normal'], densities_z['noisy'])),
            uncerts_pos = uncerts_z['adversarial'],
            uncerts_neg = np.concatenate((uncerts_z['normal'], uncerts_z['noisy'])),
            flag = flag
        )
        values[flag] = tmp_values
        labels[flag] = tmp_labels
        lrs[flag] = tmp_lr
    test_values, test_labels, test_num = evaluate_test(args, model, kdes, datatypes, nb_size)

    ## Evaluate detector
    ### evaluate on train dataset
    probs = {}
    for flag in flags:
        if args.do_test:
            probs[flag] = lrs[flag].predict_proba( test_values[flag] )[:, 1]
        else:
            probs[flag] = lrs[flag].predict_proba( values[flag] )[:, 1]
    # Compute AUC
    if args.do_test:
        n_samples = test_num
    else:
        n_samples = len(test_data)
   
    # The first 2/3 of 'probs' is the negative class (normal and noisy samples) and the last 1/3 is the positive class (adversarial samples).
    prob_datas = []
    for flag in flags:
        prob_datas.append( (probs[flag][: 2 * n_samples], probs[flag][2 * n_samples:], flag) )

    _, _, auc_score = compute_roc(
        prob_datas,
        plot=True,
        name=args.pic_name
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model',
        help="model path",
        default='../model/model_10.pth',
        type=str
    )
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'jsma' 'cw' "
             "or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.add_argument(
        '-t', '--do_test',
        help="do on test dataset",
        action='store_true',
        required=False,
    )
    parser.add_argument(
        '-u', '--uncert',
        help="uncertainty type",
        default='default',
        required=False, type=str
    )
    parser.add_argument(
        '--pic_name',
        default='result',
        required=False, type=str
    )
    parser.set_defaults(batch_size=256)
    args = parser.parse_args()
    main(args)
    



