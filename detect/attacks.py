from collections import defaultdict
import numpy as np
from tqdm import tqdm
from cleverhans.utils import other_classes
from cleverhans.utils_tf import batch_eval, model_argmax
from cleverhans.attacks import SaliencyMapMethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks

def fast_gradient_sign_method(model, X, Y, eps, test_loader=None):
    atk = torchattacks.FGSM(model, eps=eps)
    x_adv_list = []

    if test_loader is not None:
        x_adv_list = []
        for batch in test_loader:
            x = batch[0]
            y = batch[1]
            x_adv_list.append( atk(x, y) )
        X_adv = torch.cat( x_adv_list )
    else:
        X_adv = atk(X, Y)
    
    print('adv', X_adv.shape)
    X_adv = X_adv.cpu()
    return X_adv

def basic_iterative_method(model, X, Y, eps, eps_iter, test_loader=None):
    print(X.shape, Y.shape)
    atk = torchattacks.BIM(model, eps=eps, alpha=eps_iter, steps=7)
    if test_loader is not None:
        x_adv_list = []
        for batch in test_loader:
            x = batch[0]
            y = batch[1]
            x_adv_list.append( atk(x, y) )
        X_adv = torch.cat( x_adv_list )
    print('adv', X_adv.shape)
    X_adv = X_adv.cpu()
    return X_adv

'''
def basic_iterative_method(model, X, Y, eps, eps_iter, nb_iter=50, clip_min=None, clip_max=None, batch_size=256):
    # Define TF placeholders for the input and output
    x = tf.placeholder(tf.float32, shape=(None,)+X.shape[1:])
    y = tf.placeholder(tf.float32, shape=(None,)+Y.shape[1:])
    # results will hold the adversarial inputs at each iteration of BIM;
    # thus it will have shape (nb_iter, n_samples, n_rows, n_cols, n_channels)
    results = np.zeros((nb_iter, X.shape[0],) + X.shape[1:])
    # Initialize adversarial samples as the original samples, set upper and
    # lower bounds
    X_adv = X
    X_min = X_adv - eps
    X_max = X_adv + eps
    print('Running BIM iterations...')
    # "its" is a dictionary that keeps track of the iteration at which each
    # sample becomes misclassified. The default value will be (nb_iter-1), the
    # very last iteration.
    def f(val):
        return lambda: val
    its = defaultdict(f(nb_iter-1))
    # Out keeps track of which samples have already been misclassified
    out = set()
    for i in tqdm(range(nb_iter)):
        adv_x = fgsm(
            x, model(x), eps=eps_iter,
            clip_min=clip_min, clip_max=clip_max, y=y
        )
        X_adv, = batch_eval(
            sess, [x, y], [adv_x],
            [X_adv, Y], args={'batch_size': batch_size}
        )
        X_adv = np.maximum(np.minimum(X_adv, X_max), X_min)
        results[i] = X_adv
        # check misclassifieds
        predictions = model.predict_classes(X_adv, batch_size=512, verbose=0)
        misclassifieds = np.where(predictions != Y.argmax(axis=1))[0]
        for elt in misclassifieds:
            if elt not in out:
                its[elt] = i
                out.add(elt)

    return its, results
'''



def saliency_map_method(model, X, Y, theta, gamma, clip_min=None, clip_max=None):
    nb_classes = Y.shape[1]
    X_adv = np.zeros_like(X)
    # Instantiate a SaliencyMapMethod attack object
    jsma = SaliencyMapMethod(model, back='tf', sess=sess)
    jsma_params = {'theta': theta, 'gamma': gamma,
                   'clip_min': clip_min, 'clip_max': clip_max,
                   'y_target': None}
    for i in tqdm(range(len(X))):
        # Get the sample
        sample = X[i:(i+1)]
        # First, record the current class of the sample
        current_class = int(np.argmax(Y[i]))
        # Randomly choose a target class
        target_class = np.random.choice(other_classes(nb_classes,
                                                      current_class))
        # This call runs the Jacobian-based saliency map approach
        one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
        one_hot_target[0, target_class] = 1
        jsma_params['y_target'] = one_hot_target
        X_adv[i] = jsma.generate_np(sample, **jsma_params)

    return X_adv

