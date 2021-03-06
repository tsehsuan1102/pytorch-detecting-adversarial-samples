import os
import argparse
import warnings
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import KernelDensity

from detect.util import (get_data, get_model, get_noisy_samples, get_mc_predictions,get_deep_representations, score_samples, normalize,train_lr, compute_roc,get_value,AddGaussianNoise, evaluate)

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


def test_lr(lr):
    ###### get test density
    print('computing tests')
    test_densities_normal = score_samples(
        kde,
        x_test_normal_features.cpu(),
        preds_test_normal.cpu()
    )
    test_densities_noisy = score_samples(
        kdes,
        x_test_noisy_features.cpu(),
        preds_test_noisy.cpu()
    )
    test_densities_adv = score_samples(
        kdes,
        x_test_adv_features.cpu(),
        preds_test_adv.cpu()
    )
    ## Z-score the uncertainty and density values
    test_uncerts_normal_z, test_uncerts_adv_z, test_uncerts_noisy_z = normalize(
        uncerts_normal.cpu().numpy(),
        uncerts_adv.cpu().numpy(),
        uncerts_noisy.cpu().numpy()
    )
    densities_normal_z, densities_adv_z, densities_noisy_z = normalize(
        densities_normal,
        densities_adv,
        densities_noisy
    )







def main(args):
    flags = ['dense', 'uncert', 'combine']


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
            #transforms.Normalize(mean = (0.4914, 4822, 4465), std = (0.247, 0.243, 0.261) ),
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
    picked_train_data       = []
    picked_train_data_noisy = []
    picked_train_data_adv   = []
    for i, (b, y_tmp) in enumerate(zip(inds_correct, Y_train)):
        if b == 1:
            picked_train_data.append( (train_data[i][0], y_tmp) )
            picked_train_data_noisy.append( (train_noisy[i][0], y_tmp) )
            picked_train_data_adv.append( (X_train_adv[i], y_tmp) )
        else:
            continue
    picked_train_loader = DataLoader(
        dataset = picked_train_data,
        batch_size = args.batch_size
    )
    picked_train_noisy_loader = DataLoader(
        dataset = picked_train_data_noisy,
        batch_size = args.batch_size
    )
    picked_train_adv_loader = DataLoader(
        dataset = picked_train_data_adv,
        batch_size = args.batch_size
    )
    


    #######################################
    ## Get Bayesian uncertainty scores
    nb_size = 5
    print('Getting Monte Carlo dropout variance predictions...')
    uncerts_normal  = get_mc_predictions(model, picked_train_loader,         nb_iter=nb_size)#, method='entropy')
    uncerts_noisy   = get_mc_predictions(model, picked_train_noisy_loader,   nb_iter=nb_size)#, method='entropy')
    uncerts_adv     = get_mc_predictions(model, picked_train_adv_loader,     nb_iter=nb_size)#, method='entropy')
    print(uncerts_normal.shape)
    print(uncerts_noisy.shape)
    print(uncerts_adv.shape)
    
    ## Get KDE scores
    # Get deep feature representations
    print('Getting deep feature representations...')
    x_train_features         = get_deep_representations(model, train_loader              , args.dataset)
    x_train_normal_features  = get_deep_representations(model, picked_train_loader        , args.dataset)
    x_train_noisy_features   = get_deep_representations(model, picked_train_noisy_loader  , args.dataset)
    x_train_adv_features     = get_deep_representations(model, picked_train_adv_loader    , args.dataset)
    print('Shape')
    print(x_train_features.shape)
    print(x_train_normal_features.shape)
    print(x_train_noisy_features.shape)
    print(x_train_adv_features.shape)

    class_num = 10
    Y_train_label   = [ tmp[1] for tmp in train_data ]
    Y_train_label   = np.array(Y_train_label)
    Y_train         = np.zeros((len(Y_train_label), class_num))
    Y_train[ np.arange(Y_train_label.size), Y_train_label ] = 1

    # Train one KDE per class
    print('Training KDEs...')
    class_inds = {}
    for i in range(class_num):
        #class_inds[i] = np.where(Y_train.argmax(axis=1) == i)[0]
        class_inds[i] = np.where(Y_train_label == i)[0]
        print('class_inds[', i, ']: ', class_inds[i].size )

    kdes = {}
    warnings.warn("Using pre-set kernel bandwidths that were determined "
                  "optimal for the specific CNN models of the paper. If you've "
                  "changed your model, you'll need to re-optimize the bandwidth.")
    for i in range(class_num):
        kdes[i] = KernelDensity(kernel='gaussian', bandwidth=BANDWIDTHS[args.dataset]).fit( x_train_features.cpu().numpy()[class_inds[i]] )
    #print(kdes)

    # Get model predictions
    print('Computing model predictions...')
    data_loaders = [ picked_train_loader,
        picked_train_noisy_loader,
        picked_train_adv_loader
    ]
    preds = []
    for now_loader in data_loaders:
        with torch.no_grad():
            tmp_result = []
            for batch in now_loader:
                x = batch[0].to(device)
                pred = model(x)
                tmp_result.append(pred.detach().cpu())
            preds.append( torch.cat(tmp_result) )
    preds_train_normal  = torch.argmax(preds[0], dim=1)
    preds_train_noisy   = torch.argmax(preds[1], dim=1)
    preds_train_adv     = torch.argmax(preds[2], dim=1)
    #print(preds_train_normal)

    # Get density estimates
    ###### get test density
    print('computing densities...')
    densities_normal = score_samples(
        kdes,                           
        x_train_normal_features.cpu(),
        preds_train_normal.cpu()
    )
    densities_noisy = score_samples(
        kdes,
        x_train_noisy_features.cpu(),
        preds_train_noisy.cpu()
    )
    densities_adv = score_samples(
        kdes,
        x_train_adv_features.cpu(),
        preds_train_adv.cpu()
    )

    ## Z-score the uncertainty and density values
    uncerts_normal_z, uncerts_adv_z, uncerts_noisy_z = normalize(
        uncerts_normal.cpu().numpy(),
        uncerts_adv.cpu().numpy(),
        uncerts_noisy.cpu().numpy()
    )
    densities_normal_z, densities_adv_z, densities_noisy_z = normalize(
        densities_normal,
        densities_adv,
        densities_noisy
    )
    print('.......mean,,,,,,,,,,')
    print(densities_normal_z.mean())
    print(densities_adv_z.mean())
    print(densities_noisy_z.mean())

    ## Build detector
    ### combine
    values_combine, labels_combine, lr_combine = train_lr(
        densities_pos = densities_adv_z,
        densities_neg = np.concatenate((densities_normal_z, densities_noisy_z)),
        uncerts_pos = uncerts_adv_z,
        uncerts_neg = np.concatenate((uncerts_normal_z, uncerts_noisy_z)),
        flag = 'combine'
    )

    ## Build detector
    ### dense, uncert, combine
    flags = ['dense', 'uncert', 'combine']
    values  = {}
    labels  = {}
    lrs     = {}
    for flag in flags:
        tmp_values, tmp_labels, tmp_lr = train_lr(
            densities_pos = densities_adv_z,
            densities_neg = np.concatenate((densities_normal_z, densities_noisy_z)),
            uncerts_pos = uncerts_adv_z,
            uncerts_neg = np.concatenate((uncerts_normal_z, uncerts_noisy_z)),
            flag = flag
        )
        values[flag] = tmp_values
        labels[flag] = tmp_labels
        lrs[flag] = tmp_lr


    ## Evaluate detector
    # Compute logistic regression model predictions
    ### evaluate on train dataset
    probs = {}
    for flag in flags:
        probs[flag] = lrs[flag].predict_proba( values[flag] )[:, 1]
    #probs_combine   = lrs['combine'].predict_proba(values['combine'])[:, 1]
    #probs_dense     = lr_dense.predict_proba(values['dense'])[:, 1]
    #probs_uncert    = lr_uncert.predict_proba(values['uncert'])[:, 1]
    
    # Compute AUC
    n_samples = len(picked_train_data)
    # The first 2/3 of 'probs' is the negative class (normal and noisy samples) and the last 1/3 is the positive class (adversarial samples).
    prob_datas = []
    for flag in flags:
        prob_datas.append( (probs[flag][: 2 * n_samples], probs[flag][2 * n_samples:], flag) )

    _, _, auc_score = compute_roc(
        prob_datas,
        plot=True
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
    parser.set_defaults(batch_size=256)
    args = parser.parse_args()
    main(args)
    



