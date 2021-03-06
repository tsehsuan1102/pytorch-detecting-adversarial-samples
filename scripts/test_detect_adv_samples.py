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
                         AddGaussianNoise, evaluate)

# Optimal KDE bandwidths that were determined from CV tuning
BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26, 'svhn': 1.00}

def main(args):
    ## assert
    assert args.dataset in ['mnist', 'cifar', 'svhn'], "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'bim', 'jsma', 'cw', 'all'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma' or 'cw'"
    assert os.path.isfile('../data/Adv_%s_%s.npy' % (args.dataset, args.attack)), \
        'adversarial sample file not found... must first craft adversarial ' \
        'samples using craft_adv_samples.py'

    print('Loading the data and model...')
    # Load the model
    model = get_model(args.dataset)
    model.load_state_dict(torch.load(args.model))
    model.to('cuda')
    model.eval()

    # Load the dataset
    train_data, test_data = get_data(args.dataset)
    train_loader = DataLoader(
        dataset = train_data,
        batch_size = args.batch_size,
    )

    ##### Load adversarial samples (create by crate_adv_samples.py)
    print('Loading noisy and adversarial samples...')
    X_test_adv = np.load('../data/Adv_%s_%s.npy' % (args.dataset, args.attack))
    X_test_adv = torch.from_numpy(X_test_adv)
    #train_adv   = [ (x_tmp, y_tmp[1]) for x_tmp, y_tmp in zip(X_train_adv, train_data) ]
    test_adv    = [ (x_tmp, y_tmp[1]) for x_tmp, y_tmp in zip(X_test_adv, test_data) ]
    ##### create noisy data
    noise_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.1307, ), std = (0.3781, )),
        AddGaussianNoise(0., 0.1)
    ])
    train_noisy, test_noisy = get_data(args.dataset, noise_transform)


    # Check model accuracies on each sample type
    for s_type, dataset in zip(['normal', 'noisy', 'adversarial'],
                               [test_data, test_noisy, test_adv]):
        data_loader = DataLoader(
            dataset = dataset,
            batch_size = 1,
        )
        acc = evaluate(model, data_loader)
        print("Model accuracy on the %s test set: %0.2f%%" % (s_type, 100 * acc) )
        # Compute and display average perturbation sizes
        ### TODO
        '''
        if not s_type == 'normal':
            l2_diff = np.linalg.norm(
                dataset.reshape((len(X_test), -1)) - X_test.reshape((len(X_test), -1)),
                axis=1
            ).mean()
            print("Average L-2 perturbation size of the %s test set: %0.2f" % (s_type, l2_diff))
        '''

    ### Refine the normal, noisy and adversarial sets to only include samples for which the original version was correctly classified by the model
    ### run test data and choose the data model can correctly predict

    test_loader = DataLoader(
        dataset = test_data,
        batch_size = args.batch_size
    )

    y_test_list = []
    pred_list = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to('cuda')
            y_test_list.append( batch[1] )
            pred_list.append( model(x) )

    pred_test = torch.cat(pred_list)
    Y_test = torch.cat(y_test_list).to('cuda')

    inds_correct = torch.where(Y_test == pred_test.argmax(axis=1), torch.full_like(Y_test, 1), torch.full_like(Y_test, 0)).to('cuda')
    picked_test_data       = []
    picked_test_data_noisy = []
    picked_test_data_adv   = []
    for i, (b, y_tmp) in enumerate(zip(inds_correct, Y_test)):
        if b == 1:
            picked_test_data.append( (test_data[i][0], y_tmp) )
            picked_test_data_noisy.append( (test_noisy[i][0], y_tmp) )
            picked_test_data_adv.append( (X_test_adv[i], y_tmp) )
        else:
            continue
    picked_test_loader = DataLoader(
        dataset = picked_test_data,
        batch_size = args.batch_size
    )
    picked_test_noisy_loader = DataLoader(
        dataset = picked_test_data_noisy,
        batch_size = args.batch_size
    )
    picked_test_adv_loader = DataLoader(
        dataset = picked_test_data_adv,
        batch_size = args.batch_size
    )

    #######################################
    ## Get Bayesian uncertainty scores
    nb_size = 50
    print('Getting Monte Carlo dropout variance predictions...')
    uncerts_normal  = get_mc_predictions(model, picked_test_loader,         nb_iter=nb_size)#.unsqueeze(1)
    uncerts_noisy   = get_mc_predictions(model, picked_test_noisy_loader,   nb_iter=nb_size)#.unsqueeze(1)
    uncerts_adv     = get_mc_predictions(model, picked_test_adv_loader,     nb_iter=nb_size)#.unsqueeze(1)
    
    print(uncerts_normal.shape)
    print(uncerts_noisy.shape)
    print(uncerts_adv.shape)

    ## Get KDE scores
    # Get deep feature representations
    print('Getting deep feature representations...')
    x_train_features        = get_deep_representations(model, train_loader              , args.dataset)
    x_test_normal_features  = get_deep_representations(model, picked_test_loader        , args.dataset)
    x_test_noisy_features   = get_deep_representations(model, picked_test_noisy_loader  , args.dataset)
    x_test_adv_features     = get_deep_representations(model, picked_test_adv_loader    , args.dataset)
    print(x_train_features.shape)
    print(x_test_normal_features.shape)
    print(x_test_noisy_features.shape)
    print(x_test_adv_features.shape)
    
    
    class_num = 10
    Y_train_label = [ tmp[1] for tmp in train_data ]
    Y_train_label = np.array(Y_train_label)
    Y_train = np.zeros((len(Y_train_label), class_num))
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
    data_loaders = [ picked_test_loader,
        picked_test_noisy_loader,
        picked_test_adv_loader
    ]
    preds = []
    for now_loader in data_loaders:
        with torch.no_grad():
            tmp_result = []
            for batch in now_loader:
                x = batch[0].to('cuda')
                pred = model(x)
                tmp_result.append(pred)
            preds.append( torch.cat(tmp_result) )
    preds_test_normal = torch.argmax(preds[0], dim=1)
    preds_test_noisy = torch.argmax(preds[1], dim=1)
    preds_test_adv = torch.argmax(preds[2], dim=1)
    print(preds_test_normal)

    # Get density estimates
    print('computing densities...')
    densities_normal = score_samples(
        kdes,                           
        x_test_normal_features.cpu(),   
        preds_test_normal.cpu()
    )
    densities_noisy = score_samples(
        kdes,
        x_test_noisy_features.cpu(),
        preds_test_noisy.cpu()
    )
    densities_adv = score_samples(
        kdes,
        x_test_adv_features.cpu(),
        preds_test_adv.cpu()
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
    ### dense
    values_dense, labels_dense, lr_dense = train_lr(
        densities_pos = densities_adv_z,
        densities_neg = np.concatenate((densities_normal_z, densities_noisy_z)),
        uncerts_pos = uncerts_adv_z,
        uncerts_neg = np.concatenate((uncerts_normal_z, uncerts_noisy_z)),
        flag = 'dense'
    )
    ### uncert
    values_uncert, labels_uncert, lr_uncert = train_lr(
        densities_pos = densities_adv_z,
        densities_neg = np.concatenate((densities_normal_z, densities_noisy_z)),
        uncerts_pos = uncerts_adv_z,
        uncerts_neg = np.concatenate((uncerts_normal_z, uncerts_noisy_z)),
        flag = 'uncert'
    )


    ## Evaluate detector
    # Compute logistic regression model predictions
    probs_combine   = lr_combine.predict_proba(values_combine)[:, 1]
    probs_dense     = lr_dense.predict_proba(values_dense)[:, 1]
    probs_uncert    = lr_uncert.predict_proba(values_uncert)[:, 1]

    # Compute AUC
    n_samples = len(picked_test_data)
    # The first 2/3 of 'probs' is the negative class (normal and noisy samples),
    # and the last 1/3 is the positive class (adversarial samples).

    #probs_neg = probs[:2 * n_samples],
    #probs_pos = probs[2 * n_samples:],
    prob_datas = [
            (probs_combine[:2 * n_samples], probs_combine[2 * n_samples:], 'combine'),
            (probs_dense[:2 * n_samples],   probs_dense[2 * n_samples:], 'dense'),
            (probs_uncert[:2 * n_samples],  probs_uncert[2 * n_samples:], 'uncert')
    ]
    _, _, auc_score = compute_roc(
        prob_datas,
        plot=True
    )


    '''
    print('Detector ROC-AUC score: %0.4f' % auc_score)
    threshold = 0.5
    final_accu = 0
    for i in probs[:2*n_samples]:
        if i < threshold:
            final_accu += 1
    for i in probs[2*n_samples:]:
        if i > threshold:
            final_accu += 1
    print('Final: ', final_accu, n_samples*3, final_accu / (n_samples*3))
    '''


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
    





