import torch

def compute_landmarks_tr(embeddings, target, prev_landmarks=None, tau=0): # tau=0.2
    """Computing landmarks of each class in the labeled meta-dataset. Landmark is a closed form solution of 
    minimizing distance to the mean and maximizing distance to other landmarks. If tau=0, landmarks are 
    just mean of data points.
    embeddings: embeddings of the labeled dataset
    target: labels in the labeled dataset
    prev_landmarks: landmarks from previous iteration
    tau: regularizer for inter- and intra-cluster distance
    """
    
    uniq = torch.unique(target, sorted=True)
    class_idxs = list(map(lambda c: target.eq(c).nonzero(), uniq))
    
    #landmarks_mean = prev_landmarks
    #landmarks_mean[uniq] = torch.stack([embeddings[idx_class].mean(0) for idx_class in class_idxs]).squeeze()
    
    if prev_landmarks is None or tau==0:
        landmarks_mean = torch.stack([embeddings[idx_class].mean(0) for idx_class in class_idxs]).squeeze()
        return landmarks_mean
    else:
        landmarks_mean = prev_landmarks
        landmarks_mean[uniq] = torch.stack([embeddings[idx_class].mean(0) for idx_class in class_idxs]).squeeze()
        landmarks = 0.95 * prev_landmarks + 0.05 * landmarks_mean
    
    # suma = prev_landmarks.sum(0)
    # nlndmk = prev_landmarks.shape[0]
    # lndmk_dist_part = (tau/(nlndmk-1))*torch.stack([suma-p for p in prev_landmarks])
    # landmarks = 1/(1-tau)*(landmarks_mean-lndmk_dist_part)
    
        return landmarks

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def loss_cal(encoded, prototypes, target):
    """Calculate loss.
    """
    
    uniq = torch.unique(target, sorted=True)
    class_idxs = list(map(lambda c: target.eq(c).nonzero(), uniq))
    
    # prepare targets so they start from 0,1
    for idx,v in enumerate(uniq):
        target[target==v]=idx
    
    dists = euclidean_dist(encoded, prototypes)

    loss_val = torch.stack([dists[idx_example, idx_proto].mean(0) for idx_proto,idx_example in enumerate(class_idxs)]).mean()
    #loss_val1 = loss_val1/len(embeddings) 
    y_hat = torch.max(-dists,1)[1]
        
    acc_val = y_hat.eq(target.squeeze()).float().mean()    
        
    return loss_val, acc_val

def eval_loss(latent, label):
    landmarks = compute_landmarks_tr(latent, label)
    return loss_cal(latent, landmarks, label)



def init_landmarks(n_clusters, latent):
    """Initialization of landmarks of the labeled and unlabeled meta-dataset.
    nclusters: number of expected clusters in the unlabeled meta-dataset
    tr_load: data loader for labeled meta-dataset
    test_load: data loader for unlabeled meta-dataset
    """
    lndmk_test = [torch.zeros(size=(1, 32), requires_grad=True) 
                       for _ in range(n_clusters)]
    kmeans_init_test = init_step(latent, n_clusters=n_clusters)
    with torch.no_grad():
        [lndmk_test[i].copy_(kmeans_init_test[i,:]) for i in range(kmeans_init_test.shape[0])]
    return lndmk_test

def set_score(score, y_true, y_pred, scoring):
    labels=list(set(y_true))
    #print("difference", set(y_true) - set(y_pred))
    
    for metric in scoring:
        if metric=='accuracy':
            score[metric] = round(metrics.accuracy_score(y_true, y_pred), 4)
        elif metric=='precision':
            score[metric] = round(metrics.precision_score(y_true, y_pred, labels=labels, average='macro',zero_division=0), 4)
        elif metric=='recall':
            score[metric] = round(metrics.recall_score(y_true, y_pred, labels=labels, average='macro'), 4)
        elif metric=='f1_score':
            score[metric] = round(metrics.f1_score(y_true, y_pred, labels=labels, average='macro'), 4)
        elif metric=='pre_mi':
            score[metric] = round(metrics.precision_score(y_true, y_pred, labels=labels, average='micro',zero_division=0), 4)
        elif metric=='rec_mi':
            score[metric] = round(metrics.recall_score(y_true, y_pred, labels=labels, average='micro'), 4)
        elif metric=='f1_mi':
            score[metric] = round(metrics.f1_score(y_true, y_pred, labels=labels, average='micro'), 4)
        elif metric=='nmi':
            score[metric] = round(metrics.normalized_mutual_info_score(y_true, y_pred), 4)
        elif metric=='adj_mi':
            score[metric] = round(metrics.adjusted_mutual_info_score(y_true, y_pred), 4)
        elif metric=='adj_rand':
            score[metric] = round(metrics.adjusted_rand_score(y_true, y_pred), 4)
                                
def hungarian_match(y_true, y_pred):
    """Matches predicted labels to original using hungarian algorithm."""
    
    y_true = adjust_range(y_true)
    y_pred = adjust_range(y_pred)
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    # Confusion matrix.
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(-w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    d = {i:j for i, j in ind}
    y_pred = np.array([d[v] for v in y_pred])
    
    return y_true, y_pred

def adjust_range(y):
    """Assures that the range of indices if from 0 to n-1."""
    y = np.array(y, dtype=np.int64)
    val_set = set(y)
    mapping = {val:i for  i,val in enumerate(val_set)}
    y = np.array([mapping[val] for val in y], dtype=np.int64)
    return y

def compute_scores(y_true, y_pred, scoring={'accuracy','precision','recall','pre_mi','rec_mi','f1_mi','nmi',
                                                'adj_rand','f1_score','adj_mi'}):
    #y_true = y_true.cpu().numpy()
    #y_pred = y_pred.cpu().numpy()
    
    score = {}
    y_true, y_pred = hungarian_match(y_true, y_pred)
    set_score(score, y_true, y_pred, scoring)
        
    return score

def assign_labels(latent, landmk_test, label):
    """Assigning cluster labels to the unlabeled meta-dataset.
    test_iter: iterator over unlabeled dataset
    landmk_test: landmarks in the unlabeled dataset
    evaluation mode: computes clustering metrics if True
    """
    dists = euclidean_dist(latent, landmk_test)
    y_pred = torch.min(dists, 1)[1]

    scores = compute_scores(label, y_pred)
    return scores

def eval_cluster(latent, label):
    landmk_test = init_landmarks(50, latent)
    score = assign_labels(latent, torch.stack(landmk_test).squeeze(), label)
    return score
