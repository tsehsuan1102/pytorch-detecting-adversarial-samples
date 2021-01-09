import os
import multiprocessing as mp
from subprocess import call
import torch
import warnings
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale
import torchvision
from pytorchcv.model_provider import get_model as ptcv_get_model

# Gaussian noise scale sizes that were determined so that the average
# L-2 perturbation size is equal to that of the adversarial samples
STDEVS = {
    'mnist': {'fgsm': 0.310, 'bim-a': 0.128, 'bim-b': 0.265},
    'cifar': {'fgsm': 0.050, 'bim-a': 0.009, 'bim-b': 0.039},
    'svhn': {'fgsm': 0.132, 'bim-a': 0.015, 'bim-b': 0.122}
}

# Set random seed
np.random.seed(0)
datadir = '/tmp2/SPML/datasets'

def evaluate(model, data_loader):
    ## TODO batch calculation
    model.eval()
    acc = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to('cuda')
            truth = batch[1]
            pred = model(x)
            batch_sz = x.shape[0]

            for i in range(batch_sz):
                if torch.argmax(pred[i], dim=0) == truth[i]:
                    acc += 1
                total += 1
    return acc / total


def get_data(dataset='mnist', transform=None):
    ## torchvision.datasets
    assert dataset in ['mnist', 'cifar', 'svhn'], \
       "dataset parameter must be either 'mnist' 'cifar' or 'svhn'"

    if transform is None:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean = (0.1307, ), std = (0.3081, )),
        ])
    else:
        transform = transform

    if dataset == 'mnist':
        train_data = torchvision.datasets.MNIST(
            root=datadir,
            download=True,
            train=True,
            #train=False,
            transform=transform,
        )
        test_data = torchvision.datasets.MNIST(
            root=datadir,
            download=True,
            train=False,
            transform=transform,
        )
        return train_data, test_data

    elif dataset == 'cifar':
        train_data = torchvision.datasets.CIFAR10(
            root=datadir,
            download=True,
            train=True,
            transform=transform
        )
        test_data = torchvision.datasets.CIFAR10(
            root=datadir,
            download=True,
            train=False,
            transform=transform
        )
        return train_data, test_data

    '''
    else:
        train_data = torchvision.datasets.SVHN(
            root=datadir,
            download=True,
            train=True,
            transform=torchvision.transforms.ToTensor(),
        )

        train = sio.loadmat('../data/svhn_train.mat')
        test = sio.loadmat('../data/svhn_test.mat')
        
        X_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
        X_test = np.transpose(test['X'], axes=[3, 0, 1, 2])
        
        # reshape (n_samples, 1) to (n_samples,) and change 1-index
        # to 0-index
        
        y_train = np.reshape(train['y'], (-1,)) - 1
        y_test = np.reshape(test['y'], (-1,)) - 1

    # cast pixels to floats, normalize to [0, 1] range
    #X_train = X_train.astype('float32')
    #X_test = X_test.astype('float32')
    #X_train /= 255
    #X_test /= 255

    # one-hot-encode the labels
    #Y_train = np_utils.to_categorical(y_train, 10)
    #Y_test = np_utils.to_categorical(y_test, 10)

    #print(X_train.shape)
    #print(Y_train.shape)
    #print(X_test.shape)
    #print(Y_test.shape)

    #return X_train, Y_train, X_test, Y_test
    '''


def get_model(dataset='mnist'):
    assert dataset in ['mnist', 'cifar', 'svhn'], \
        "dataset parameter must be either 'mnist' 'cifar' or 'svhn'"
    
    if dataset == 'mnist':
        # MNIST model
        model = torch.nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d( (2, 2) ),
            nn.Dropout(p = 0.5),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(128, 10),
            nn.Softmax(),
        )
        print(model)
        return model
        '''
        layers = [
            Conv2D(64, (3, 3), padding='valid', input_shape=(28, 28, 1)),
            Activation('relu'),
            Conv2D(64, (3, 3)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),
            Flatten(),
            Dense(128),
            Activation('relu'),
            Dropout(0.5),
            Dense(10),
            Activation('softmax')
        ]
        '''
    elif dataset == 'cifar':
        #model = torchvision.models.resnet18(pretrained=False)
        model = ptcv_get_model('resnet20_cifar10', pretrained=True)
        print(model)
        return model

    '''
    elif dataset == 'cifar':
        # CIFAR-10 model
        layers = [
            Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
            Activation('relu'),
            Conv2D(32, (3, 3), padding='same'),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), padding='same'),
            Activation('relu'),
            Conv2D(64, (3, 3), padding='same'),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), padding='same'),
            Activation('relu'),
            Conv2D(128, (3, 3), padding='same'),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
            Activation('relu'),
            Dropout(0.5),
            Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
            Activation('relu'),
            Dropout(0.5),
            Dense(10),
            Activation('softmax')
        ]

    else:
        # SVHN model
        layers = [
            Conv2D(64, (3, 3), padding='valid', input_shape=(32, 32, 3)),
            Activation('relu'),
            Conv2D(64, (3, 3)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),
            Flatten(),
            Dense(512),
            Activation('relu'),
            Dropout(0.5),
            Dense(128),
            Activation('relu'),
            Dropout(0.5),
            Dense(10),
            Activation('softmax')
        ]

    model = Sequential()
    for layer in layers:
        model.add(layer)
    print(model)
    return model
    '''

def flip(x, nb_diff):
    """
    Helper function for get_noisy_samples
    :param x:
    :param nb_diff:
    :return:
    """
    original_shape = x.shape

    x = np.copy(np.reshape(x, (-1,)))
    candidate_inds = np.where(x < 0.99)[0]
    assert candidate_inds.shape[0] >= nb_diff
    inds = np.random.choice(candidate_inds, nb_diff)
    x[inds] = 1.

    return np.reshape(x, original_shape)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_noisy_samples(X_test, X_test_adv, dataset, attack):
    if attack in ['jsma', 'cw']:
        X_test_noisy = np.zeros_like(X_test)
        for i in range(len(X_test)):
            # Count the number of pixels that are different
            nb_diff = len(np.where(X_test[i] != X_test_adv[i])[0])
            # Randomly flip an equal number of pixels (flip means move to max
            # value of 1)
            X_test_noisy[i] = flip(X_test[i], nb_diff)

    else:
        warnings.warn("Using pre-set Gaussian scale sizes to craft noisy samples. If you've altered the eps/eps-iter parameters "
                      "of the attacks used, you'll need to update these. In the future, scale sizes will be inferred automatically "
                      "from the adversarial samples.")
        # Add Gaussian noise to the samples
        X_test_noisy = np.minimum(
            np.maximum(
                X_test + np.random.normal(loc=0, scale=STDEVS[dataset][attack], size=X_test.shape), 0 ), 1
        )

    return X_test_noisy

"""
def get_mc_predictions(model, X, nb_iter=50, batch_size=256):
    output_dim = model.layers[-1].output.shape[-1].value
    get_output = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[-1].output]
    )


    def predict():
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        output = np.zeros(shape=(len(X), output_dim))
        for i in range(n_batches):
            #X[(i*batch_size) : ((i+1)*batch_size) ]

            #output[ (i*batch_size) : ((i+1)*batch_size) ] = 

            output[i * batch_size:(i + 1) * batch_size] = \
                get_output([X[i * batch_size:(i + 1) * batch_size], 1])[0]
        return output

    preds_mc = []
    for i in tqdm(range(nb_iter)):
        preds_mc.append(predict())

    return np.asarray(preds_mc)
"""

def get_mc_predictions(model, dataloader, nb_iter=50, method = "default"):
    is_training = model.training
    
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

    preds_mc = []
    trange_mc = tqdm(range(nb_iter), total = nb_iter, desc = "MC Epoch")

    for _ in trange_mc:
        outputs = []
        trange_loader = tqdm(dataloader, total = len(dataloader), desc = "Prediction", leave=False)

        for batch in trange_loader:
            #x = torch.stack([[batch_data[0] for batch_data in batch]]).to("cuda")
            x = batch[0].to('cuda')

            with torch.no_grad():
                outputs.append(model(x))

        preds_mc.append(torch.cat(outputs))

    if is_training:
        model.train()
    else:
        model.eval()

    if method == "default":
        preds_mc = torch.stack(preds_mc).double()
        mc_dot_mean = (preds_mc * preds_mc).sum(2).mean(0)
        mc_mean_dot = (preds_mc.mean(0) * preds_mc.mean(0)).sum(1)
        uncertainty = mc_dot_mean - mc_mean_dot
    elif method == "entropy":
        preds_mc = torch.stack(preds_mc).double()
        mc_mean = preds_mc.mean(0)
        uncertainty = -1.0 * (mc_mean * torch.log(mc_mean)).sum(1)
    else:
        raise ValueError("只有paper上的那種跟entropy啦幹")

    return uncertainty

"""
eef get_deep_representations(model, X, batch_size=256):
    TODO
    :param model:
    :param X:
    :param batch_size:
    :return:
    
    # last hidden layer is always at index -4
    output_dim = model.layers[-4].output.shape[-1].value
    get_encoding = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[-4].output]
    )

    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    output = np.zeros(shape=(len(X), output_dim))
    for i in range(n_batches):
        output[i * batch_size:(i + 1) * batch_size] = \
            get_encoding([X[i * batch_size:(i + 1) * batch_size], 0])[0]

    return output
"""
def get_deep_representations(model, dataloader, dataset_type):
    trange = tqdm(dataloader, total = len(dataloader), desc = "Deep Representation")
    representations = []

    for batch in trange:
        x = batch[0].to('cuda')

        with torch.no_grad():
            if dataset_type == "mnist":
                representations.append( model[: -3](x).detach().cpu() )

            elif dataset_type == "cifar":
                representations.append(model._modules["features"](x).squeeze().detach().cpu() )

            elif dataset_type == "svhn":
                raise NotImplementedError("幹你娘")
            else:
                raise ValueError("沒這個選項啦幹(只有mnist跟cifar_10)")

    return torch.cat(representations)


def score_point(tup):
    x, kde = tup
    return kde.score_samples(np.reshape(x, (1, -1)))[0]


def score_samples(kdes, samples, preds, n_jobs=None):
    #mp.set_start_method('spawn')
    if n_jobs is not None:
        p = mp.Pool(n_jobs)
    else:
        p = mp.Pool()

    results = np.asarray(
        list(
            map(
            score_point, [(x, kdes[i.item()]) for x, i in zip(samples, preds)]
        ))
    )
    print('results', results.size)
    print('results', results.mean())
    p.close()
    p.join()

    return results


def normalize(normal, adv, noisy):
    """
    TODO
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    #print(normal)
    #print(type(normal))

    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv, noisy)))

    return total[:n_samples], total[n_samples:2*n_samples], total[2*n_samples:]


def train_lr(densities_pos, densities_neg, uncerts_pos, uncerts_neg, flag='combine'):
    ### uncert
    if flag == 'uncert':
        values_neg = uncerts_neg.reshape((1, -1)).transpose([1, 0])
        values_pos = uncerts_pos.reshape((1, -1)).transpose([1, 0])
    ### dense
    if flag == 'dense':
        values_neg = densities_neg.reshape((1, -1)).transpose([1, 0])
        values_pos = densities_pos.reshape((1, -1)).transpose([1, 0])
    ### combine
    if flag == 'combine':
        values_neg = np.concatenate(
            (densities_neg.reshape((1, -1)),
             uncerts_neg.reshape((1, -1))),
            axis=0).transpose([1, 0])
        values_pos = np.concatenate(
            (densities_pos.reshape((1, -1)),
             uncerts_pos.reshape((1, -1))),
            axis=0).transpose([1, 0])
    
    values = np.concatenate((values_neg, values_pos))
    #labels = np.concatenate(
    #    (np.zeros_like(values_neg), np.ones_like(values_pos))
    #)
    labels = np.concatenate(
        (np.zeros_like(densities_neg), np.ones_like(densities_pos))
    )
    lr = LogisticRegressionCV(n_jobs=-1).fit(values, labels)

    return values, labels, lr


#def compute_roc(probs_neg, probs_pos, plot=False):
def compute_roc(probs_data, plot=False):
    
    if plot:
        plt.figure(figsize=(7, 6))

    for probs_neg, probs_pos, name in probs_data:
        probs = np.concatenate((probs_neg, probs_pos))
        labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_score = auc(fpr, tpr)
        print(name, ' auc_score: ', auc_score)
        if plot:
            plt.plot(fpr, tpr, label=name)#label='ROC (AUC = %0.4f)' % auc_score)

    if plot:    
        plt.legend(loc='lower right')#, labels=['combine', 'dense', 'uncert'])
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
         #plt.show()
        plt.savefig('./result.png')

    return fpr, tpr, auc_score



