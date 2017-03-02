from cnn_txt_model import *

import cPickle
import numpy as np
import pdb
from torch.autograd import Variable
import climate
import logging
import sys
import os

import argparse
import torch

logging = climate.get_logger(__name__)
climate.enable_default_logging()

#batch_size = 10
#llen = 10
#word_len = 300
#
#x = torch.randn(batch_size, 1, llen, word_len)
#
#model = CNN_Txt_Net(llen)
#
#output = model(Variable(x))


def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x


def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        sent.append(rev["y"])
        if rev["split"]==cv:            
            test.append(sent)        
        else:  
            train.append(sent)   

    train = np.array(train)
    train = np.asarray(train, dtype = 'int32')
    test = np.array(test)
    test = np.asarray(test, dtype = 'int32')
    return [train, test]     


def accuracy(y_pred, y, topk=(1, )):

    maxk = max(topk)
    batch_size = y.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

if __name__=="__main__":

    parser = argparse.ArgumentParser(description = 'PYTorch CNN for Sentence')
    parser.add_argument('--batch_size', type = int, default = 64, metavar = 'N',
                    help = 'input batch size for training (default 64)')
    
    parser.add_argument('--test-batch-size', type = int, default = 1000, metavar = 'N',
                    help = 'input batch size for testing (default 1000)')
    
    parser.add_argument('--epochs', type = int, default = 10, metavar = 'N',
                    help = 'number of epochs to train (default: 10)')
    
    parser.add_argument('--lr', type = float, default = 0.01, metavar = 'LR',
                    help = 'learning rate (default: 0.01)')
    
    parser.add_argument('--momentum', type = float, default = 0.9, metavar = 'M',
                    help = 'SGD momentum (default:0.5)')
    
    parser.add_argument('--no-cuda', action = 'store_true', default = False,
                    help = 'enables cuda training')
    
    parser.add_argument('--seed', type = int, default = 1, metavar = 'S',
                    help = 'random seed (default: 1)')
    
    parser.add_argument('--log-interval', type = int, default = 10, metavar = 'N',
                    help = 'how many batches to wait before logging training status')
    parser.add_argument('--word2vec_len', type = int, default = 300, metavar ='N',
                    help = 'Wor2veclen')

    parser.add_argument('--mode', type = str, default = 'static', metavar = 'MO',
                    help = 'the mode, default is static')
    
    parser.add_argument('--word_vectors', type = str, default = 'word2vec', metavar = 'WV',
                    help = 'word2vec mode')

    parser.add_argument('--save_dir', type = str, default = 'model_CNN_Txt_Net', metavar = 'M',
                    help = 'saving directory for the model')

    parser.add_argument('--max_l', type = int, default = 56,
            help = 'max length of the sentence')
    parser.add_argument('--print_every', type = int, default = 10,
            help = 'frequency of printing loss')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    logging.info("loading data...")
    x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    logging.info("data loaded!")

    mode= args.mode
    if mode=="nonstatic":
        logging.info("model architecture: CNN-non-static")
        non_static=True
    elif mode=="static":
        logging.info("model architecture: CNN-static")
        non_static=False

    if args.word_vectors=="rand":
        logging.info("using: random vectors")
        U = W2
    elif args.word_vectors=="word2vec":
        logging.info("using: word2vec vectors")
        U = W

    py_words = torch.from_numpy(U)

    batch_data = torch.LongTensor(args.batch_size, args.max_l)
    batch_lbl = torch.LongTensor(args.batch_size)


    criterion = nn.NLLLoss()

    if args.cuda:
        criterion.cuda()
        batch_data = batch_data.cuda()
        batch_lbl = batch_lbl.cuda()

    batch_data = Variable(batch_data)
    batch_lbl = Variable(batch_lbl)
    
    def weights_init(m):
        classname = m.__class__.__name__
        print classname
        logging.info('classname')
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        # elif classname.find('Linear') != -1:
        #    m.weight.data.uniform_(0.0, 1)
        #    m.bias.data.fill_(0)
    
   
    results = []
    batch_size = args.batch_size
    r = range(0,10)    


    for i in r:
        # Model.
        net_emb = nn.Embedding(len(vocab) + 1, args.word2vec_len, 0)
        net_emb.weight.data.copy_(py_words)
        net_cnn = CNN_Txt_Net(args.max_l)
        if args.cuda:
            net_emb.cuda()
            net_cnn.cuda()
        net_cnn.apply(weights_init)
        opt_emb = torch.optim.SGD(net_emb.parameters(), args.lr, momentum = args.momentum)
        

        opt_cnn = torch.optim.SGD(net_cnn.parameters(), args.lr, momentum = args.momentum)

        print(net_cnn)
 
        # reinitialize the parameters.
        net_cnn.apply(weights_init)
        net_emb.weight.data.copy_(py_words)

        datasets = make_idx_data_cv(revs, word_idx_map, i, max_l=args.max_l,k= args.word2vec_len, filter_h=5)
        if datasets[0].shape[0] % batch_size > 0:
            extra_data_num = batch_size - datasets[0].shape[0] % batch_size
            train_set = np.random.permutation(datasets[0])   
            extra_data = train_set[:extra_data_num]
            new_data=np.append(datasets[0],extra_data,axis=0)
        else:
            new_data = datasets[0]

        new_data = np.random.permutation(new_data)
        n_batches = new_data.shape[0]/batch_size
        n_train_batches = int(np.round(n_batches*0.9))
        #divide train set into train/val sets 
        
        #test_dataset = datasets[1][:,:args.max_l]
        test_dataset = np.random.permutation(datasets[1])

        test_set_x = torch.from_numpy(test_dataset[:,:args.max_l])
        test_set_y = torch.from_numpy(np.asarray(test_dataset[:,-1],"int32"))

        n_test_batches = test_set_x.size(0) / batch_size
        train_set = new_data[:n_train_batches*batch_size,:]
        val_set = new_data[n_train_batches*batch_size:,:]     
        train_set_x, train_set_y = torch.from_numpy(train_set[:,:args.max_l]), torch.from_numpy(train_set[:,-1])
        val_set_x, val_set_y = torch.from_numpy(val_set[:,:args.max_l]), torch.from_numpy(val_set[:,-1])
        n_val_batches = n_batches - n_train_batches

        net_emb.train()
        net_cnn.train()
        save_cnn_fn = os.path.join(args.save_dir, "cnn_r_" + str(i) + "_epochs_" + str(args.epochs))
        save_emb_fn = os.path.join(args.save_dir, "emb_r_" + str(i) + "_epochs_" + str(args.epochs))

        logging.info("saving to %s", save_cnn_fn)
        logging.info("saving to %s", save_emb_fn)

        for epoch in range(args.epochs):
        #for epoch in range(1):
            for idx,mini_batch_idx in enumerate(np.random.permutation(range(n_train_batches))):
            #for idx,mini_batch_idx in enumerate(np.random.permutation(range(2))):

                if args.mode == 'static':
                    net_emb.zero_grad()
                net_cnn.zero_grad()
                
                x = train_set_x[mini_batch_idx * batch_size :(mini_batch_idx + 1) * batch_size]
                y = train_set_y[mini_batch_idx * batch_size :(mini_batch_idx + 1) * batch_size]
                batch_data.data.resize_(x.size()).copy_(x)
                batch_lbl.data.resize_(y.size()).copy_(y)
                
                if args.cuda:
                    batch_data.cuda()
                    batch_lbl.cuda()

                out = net_emb(batch_data)
                si = out.size()
                out = out.view(si[0], 1, si[1], si[2])
                out = net_cnn(out)
                loss = criterion(out, batch_lbl)
                
                loss.backward()

                opt_cnn.step()
                if args.mode == 'static':
                    opt_emb.step()
                
                if idx % args.print_every == 0:
                    logging.info('i = %d, epoch = %d, mini-b = %d/%d, loss = %.4f', i, epoch, idx, n_train_batches, loss.data[0])
            # After each epoch, lets do a validation.
            #for idx, mini_batch_idx in enumerate(range(2)):
            for idx, mini_batch_idx in enumerate(range(n_val_batches)):
                net_emb.eval()
                net_cnn.eval()

                x = val_set_x[mini_batch_idx * batch_size : (mini_batch_idx + 1) * batch_size]
                y = val_set_y[mini_batch_idx * batch_size : (mini_batch_idx + 1) * batch_size]
                batch_data.data.resize_(x.size()).copy_(x)
                batch_lbl.data.resize_(y.size()).copy_(y)

                if args.cuda:
                    batch_data.cuda()
                    batch_lbl.cuda()
                out = net_emb(batch_data)
                si = out.size()
                out = out.view(si[0], 1, si[1], si[2])
                out = net_cnn(out)
                loss = criterion(out, batch_lbl)
                ac = accuracy(out, batch_lbl)
                logging.info('i = %d, loss = %.3f, accuracy = %.3f', i, loss.data[0], ac[0].data[0])

        for idx, mini_batch_idx in enumerate(range(n_test_batches)):
            net_emb.eval()
            net_cnn.eval()

            x = test_set_x[mini_batch_idx * batch_size : (mini_batch_idx + 1) * batch_size]
            y = test_set_y[mini_batch_idx * batch_size : (mini_batch_idx + 1) * batch_size]
            batch_data.data.resize_(x.size()).copy_(x)
            batch_lbl.data.resize_(y.size()).copy_(y)

            if args.cuda:
                batch_data.cuda()
                batch_lbl.cuda()
            out = net_emb(batch_data)
            si = out.size()
            out = out.view(si[0], 1, si[1], si[2])
            out = net_cnn(out)
            loss = criterion(out, batch_lbl)
            ac = accuracy(out, batch_lbl)
            logging.info('testing, i = %d, loss = %.3f, accuracy = %.3f',i, loss.data[0], ac[0].data[0])

        torch.save(net_cnn.state_dict(), save_cnn_fn)
        torch.save(net_emb.state_dict(), save_emb_fn)
