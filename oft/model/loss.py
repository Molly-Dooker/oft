import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt




def masked_l1_loss(input, target, mask):
    return (F.l1_loss(input, target, reduction='none') * mask.float()).sum()

def huber_loss(input, target, mask=None):
    loss = F.smooth_l1_loss(input, target, reduction='none')
    if mask is None:
        return loss.sum()
    return (loss * mask.float()).sum()


def hard_neg_mining_loss(scores, labels, neg_ratio=5):

    # Flatten tensors along the spatial dimensions
    scores = scores.flatten(2, 3)
    labels = labels.flatten(2, 3)
    count = labels.size(-1)

    # Rank negative locations by the predicted confidence
    _, inds = (scores.sigmoid() * (~labels).float()).sort(-1, descending=True)
    ordinals = torch.arange(count, out=inds.new_empty(count)).expand_as(inds)
    rank = torch.empty_like(inds)
    rank.scatter_(-1, inds, ordinals)

    # Include only positive locations + N most confident negative locations
    num_pos = labels.long().sum(dim=-1, keepdim=True)
    num_neg = (num_pos + 1) * neg_ratio
    mask = (labels | (rank < num_neg)).float()

    # Apply cross entropy loss
    return F.binary_cross_entropy_with_logits(
        scores, labels.float(), mask, reduction='sum')


def balanced_cross_entropy_loss(scores, labels):
    labels = labels.float()
    
    # Weight the loss by the relative number of positive and negative examples
    num_pos = int(labels.long().sum()) + 1
    num_neg = labels.numel() - num_pos
    weights = (num_neg - num_pos) * labels + num_pos

    # Compute cross entropy loss
    return F.binary_cross_entropy_with_logits(scores, labels, weights)


# def heatmap_loss(scores, labels, thresh=0.05, pos_weight=100):
#     labels = labels.float()
#     mask = (labels > thresh).float()
#     loss = F.l1_loss(scores, labels, reduction='none')
#     weighted = loss * (1. + (pos_weight - 1.) * mask)

#     return weighted.sum()


def heatmap_loss(heatmap, gt_heatmap, weights=[100], thresh=0.05):
    
    positives = (gt_heatmap > thresh).float()
    weights = heatmap.new(weights).view(1, -1, 1, 1)

    loss = F.l1_loss(heatmap, gt_heatmap, reduce=False)
    
    loss *= positives * weights + (1 - positives)
    return loss.sum()

def focal_loss(pred_heatmap, gt_heatmap, alpha=0.25, gamma=2.0):
    """
    Focal loss for heatmaps.
    
    Args:
        pred_heatmap (torch.Tensor): Predicted heatmaps (logits) from the model.
        gt_heatmap (torch.Tensor): Ground truth heatmaps.
        alpha (float): Alpha balancing factor.
        gamma (float): Gamma focusing factor.
    """
    # 모델 출력을 확률 값으로 변환
    p = torch.sigmoid(pred_heatmap)

    # BCE Loss 계산을 위한 준비
    # p_t는 정답 레이블에 대한 모델의 예측 확률을 의미합니다.
    # gt=1일 때 p_t=p, gt=0일 때 p_t=1-p
    p_t = p * gt_heatmap + (1 - p) * (1 - gt_heatmap)
    
    # Modulating factor (1 - p_t)^gamma
    # 핵심: p_t가 1에 가까울수록(쉬운 샘플), 이 값이 0에 가까워져 loss를 줄여줌
    modulating_factor = (1.0 - p_t).pow(gamma)
    
    # Alpha-balanced factor
    # 클래스별 가중치. gt=1일 때 alpha, gt=0일 때 1-alpha
    alpha_t = alpha * gt_heatmap + (1 - alpha) * (1 - gt_heatmap)
    
    # Focal Loss 계산
    # F.binary_cross_entropy_with_logits는 sigmoid와 bce_loss를 합친 것과 같습니다.
    # 수치적으로 더 안정적입니다.
    bce_loss = F.binary_cross_entropy_with_logits(pred_heatmap, gt_heatmap, reduction='none')
    
    focal_loss = alpha_t * modulating_factor * bce_loss
    
    # 정답 위치(positive)의 개수로 나누어 정규화하거나, 전체 픽셀 수로 나눌 수 있습니다.
    # 원본 논문에서는 positive 개수로 나누는 것을 제안합니다.
    num_positives = gt_heatmap.eq(1).float().sum()
    if num_positives > 0:
        return focal_loss.sum() / num_positives
    else:
        # Positive 샘플이 없는 경우의 예외 처리
        return focal_loss.sum()

# def uncertainty_loss(logvar, sqr_dists):
#     sqr_dists = sqr_dists.clamp(min=1.+1e-6)
#     c = (1 + torch.log(sqr_dists)) / sqr_dists
#     loss = torch.log1p(logvar.exp()) / sqr_dists + torch.sigmoid(-logvar) - c
#     print('dists', float(sqr_dists.min()), float(sqr_dists.max()))
#     print('logvar', float(logvar.min()), float(logvar.max()))
#     print('loss', float(loss.min()), float(loss.max()))

#     def hook(grad):
#         print('grad', float(grad.min()), float(grad.max()), float(grad.sum()))
#     logvar.register_hook(hook)

#     return loss.mean()


def uncertainty_loss(logvar, sqr_dists):
    dists = sqr_dists + 1.
    loss = torch.exp(-logvar) + (logvar - dists.log() - 1) / dists
    print('dists', float(sqr_dists.min()), float(sqr_dists.max()))
    print('logvar', float(logvar.min()), float(logvar.max()))
    print('loss', float(loss.min()), float(loss.max()))

    def hook(grad):
        print('grad', float(grad.min()), float(grad.max()), float(grad.sum()))
    logvar.register_hook(hook)

    if (logvar > 10).any():
        raise RuntimeError()

    return loss.mean()


def compute_uncertainty(logvar, sqr_dists, min_dist):
    var = torch.exp(logvar)
    return min_dist / torch.sqrt(var) * torch.exp(
        -0.5 * (sqr_dists / logvar - 1.))

CONST = 1.1283791670955126
def log_ap_loss(logvar, sqr_dists, num_thresh=10):


    print('dists', float(sqr_dists.min()), float(sqr_dists.max()))
    print('logvar', float(logvar.min()), float(logvar.max()))

    def hook(grad):
        print('grad', float(grad.min()), float(grad.max()), float(grad.sum()))
    logvar.register_hook(hook)

    variance = torch.exp(logvar).view(-1, 1)
    stdev = torch.sqrt(variance)
    print('stdev', float(stdev.min()), float(stdev.max()))

    max_dist = math.sqrt(float(sqr_dists.max()))
    minvar, maxvar = float(stdev.min()), float(stdev.max())
    thresholds = torch.logspace(
        math.log10(1 / maxvar), math.log10(max_dist / minvar), num_thresh).type_as(stdev)
    
    print('maxdist: {:.2e} minvar: {:.2e} maxvar: {:.2e}'.format(max_dist, minvar, maxvar))
    print('thresholds {:.2e} - {:.2e}'.format(thresholds.min(), thresholds.max()))

    k_sigma = stdev * thresholds
    k_sigma_sqr = variance * thresholds ** 2
    mask = (sqr_dists.view(-1, 1) < k_sigma_sqr).float()

    erf = torch.erf(k_sigma)
    masked_erf = erf * mask
    masked_exp = stdev * torch.exp(-k_sigma_sqr) * mask

    loss = masked_exp.sum(0) * masked_erf.sum(0) / erf.sum(0)
    loss = (loss[0] + loss[-1]) / 2. + loss[1:-1].sum()
    return -torch.log(loss * CONST / len(variance))

