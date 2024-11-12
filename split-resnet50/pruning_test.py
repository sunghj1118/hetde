import torch
from torch.nn.utils import prune
import torch.nn as nn

class IWisePruningConv2DKernels(prune.BasePruningMethod):
    PRUNING_TYPE = 'global'

    def __init__(self, amount):
        self.amount = amount

    def compute_mask(self, t, default_mask):
        weight = t
        O, I, H, W = weight.shape
        # Start with the current mask (pruned areas remain pruned)
        mask = default_mask.clone() # Shape: (O, I, H, W)

        mask_Iwise = torch.sum(mask, dim=[0, 2, 3])  # Shape: (I, )
        l1_norms = torch.sum(torch.abs(weight), dim=[0, 2, 3])

        is_nonpruned = (mask_Iwise != 0) # Shape: (I, )
        

        nonpruned_l1_norms = l1_norms[is_nonpruned]
        num_nonpruned_kernels = nonpruned_l1_norms.numel() # num_nonpruned_kernels
        k = int(num_nonpruned_kernels * self.amount)
        if k == 0:
            return mask
        else:
            threshold_value = torch.kthvalue(nonpruned_l1_norms, k).values
            mask_Iwise = (l1_norms > threshold_value) & is_nonpruned
            mask = mask_Iwise.view(1, I, 1, 1).expand(O, -1, H, W).float()

        return mask


"""
입력 채널의 pruning 여부를 True/False 리스트로 반환
ex) [True, False, False] => 첫 번째 입력 채널과 곱해지는 weigth가 모두 0이라는 뜻
"""
def is_input_channel_pruned(conv: torch.nn.Conv2d):
    return (torch.sum(conv.weight_mask, dim = [0, 2, 3]) == 0).tolist()


if __name__ == '__main__':
    conv_layer = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, padding=1)
    a = 0.4  # Prune the lowest 40% of kernels
    IWisePruningConv2DKernels.apply(conv_layer, 'weight', amount=a)
    print('mask_shape:', conv_layer.weight_mask.shape)
    print('mask:', conv_layer.weight_mask)
    IWisePruningConv2DKernels.apply(conv_layer, 'weight', amount=a)
    print('mask:', conv_layer.weight_mask)

    # Find which input channels were pruned
    print('is input channel pruned:', is_input_channel_pruned(conv_layer))

    # prune.remove(conv_layer, 'weight') to remove the mask and the weight_orig while applying the mask to the weight 

    # Prune the whole model
    # for name, module in model.named_modules():
    #    if isinstance(module, nn.Conv2d): # or (and not input layer)
    #        IWisePruningConv2DKernels.apply(module, 'weight', amount=a)