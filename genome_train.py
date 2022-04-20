import  torch
import  numpy as np
from    MiniImagenet import MiniImagenet
from    torch.utils.data import DataLoader
import  argparse

from genome_meta import Meta
from genome_loader import ContigData

def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    random.seed(222)

    # print(args)

    dataset = ContigData('/data/yunxiang/data/', mode='train', batchsz=10000, n_way=args.n_way, 
                         k_shot=args.k_spt, tasks=['airways', 'gi', 'oral', 'skin', 'urog'])
    dataset_test = ContigData('/data/yunxiang/data/', mode='train', batchsz=100, n_way=args.n_way, 
                         k_shot=args.k_qry, tasks=['airways'])

    nsamples = dataset.rpkms.shape[1]
    device = torch.device('cuda')
    maml = Meta(args, nsamples).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(dataset, 1, shuffle=True, num_workers=1, pin_memory=True)

        for step, (tnf_spt, rpkm_spt, y_spt) in enumerate(db):

            tnf_spt, rpkm_spt, y_spt = tnf_spt.to(device), rpkm_spt.to(device), y_spt.to(device)

            losses_q = maml(tnf_spt, rpkm_spt, y_spt)

            if step % 30 == 0:
                print('step:', step, '\ttraining acc:', losses_q)

            if step % 100 == 0:  # evaluation
                db_test = DataLoader(dataset_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                scores = {'accuracy':0,'precision':0,'recall':0,'pre_mi':0,'rec_mi':0,'f1_mi':0,'nmi':0,
                                                'adj_rand':0,'f1_score':0,'adj_mi':0}

                for tnf_qry, rpkm_qry, y_qry in db_test:
                    tnf_qry, rpkm_qry, y_qry = tnf_qry.squeeze(0).to(device), rpkm_qry.squeeze(0).to(device), \
                                                 y_qry.squeeze(0).to(device)

                    score = maml.finetunning(tnf_qry, rpkm_qry, y_qry)
                    for key, item in scores.items():
                        scores[key] += score[key]

                # [b, update_step+1]
                for key, item in scores.items():
                    scores[key] /= 100
                # accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Acc {}, F1_score {}, Precision {}, Recall {}, F1_mi {}, Pre_mi {}, Rec_mi {}, NMI {}, Adj_Rand {}, Adj_MI {}'.format( 
                    scores['accuracy'],scores['f1_score'],scores['precision'],scores['recall'],scores['nmi'],scores['pre_mi'],scores['rec_mi'],scores['f1_mi'],
                    scores['adj_rand'],scores['adj_mi']))


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=50)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
