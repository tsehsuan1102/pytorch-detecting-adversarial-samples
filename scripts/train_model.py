import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from detect.util import get_data, get_model, evaluate


def main(args):
    print(args)

    assert args.dataset in ['mnist', 'cifar', 'svhn'], \
        "dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    print('Data set: %s' % args.dataset)
    
    train_data, test_data = get_data(args.dataset)
    train_loader = DataLoader(
        dataset = train_data,
        batch_size = args.batch_size,
    )
    test_loader = DataLoader(
        dataset = test_data,
        batch_size = args.batch_size
    )

    model = get_model(args.dataset)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    loss_criterien = torch.nn.CrossEntropyLoss()
    running_loss = 0
    
    for i_epoch in range(args.epochs):
        model.to('cuda')
        model.train()

        print('training %d epochs' % (i_epoch) )

        pbar = tqdm(train_loader)
        ## train
        for batch_data in pbar:
            optimizer.zero_grad()
            x = batch_data[0].to('cuda')
            y = batch_data[1].to('cuda')
            batch_sz = x.shape[0]
            
            pred = model(x)
            target = torch.tensor(y)

            loss = loss_criterien(pred, target)
            #print(loss)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            pbar.set_description('loss: %f' % (loss) )

        ## eval 
        #test_pbar = tqdm(test_loader)
        acc = evaluate(model, test_loader)
        '''
        for batch_data in test_pbar:
            with torch.no_grad():
                x = batch_data[0].to('cuda')
                y = batch_data[1].to('cuda')
                batch_sz = x.shape[0]
                print(x.shape)

                pred = model(x)
                acc = evaluate()
                if torch.argmax(pred, dim=1) == y:
                    acc += 1
                total += 1
        '''
        print('DEV accuracy: ', acc)   
    
    model_dir = '../model/'
    path = model_dir + args.dataset + '_' + str(args.epochs) + '.pth'
    torch.save(model.state_dict(), path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.add_argument(
        '-lr', '--learning_rate',
        help='learning rate', dest='lr',
        default=1e-3, type=float
    )
    parser.set_defaults(epochs=20)
    parser.set_defaults(batch_size=128)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    main(args)
    







