import argparse, os
import numpy as np
import torch
from theconf import Config as C, ConfigArgumentParser
from FastAutoAugment.controller import Controller
from FastAutoAugment.data import get_dataloaders, Augmentation, get_custom_dataloaders, CutoutDefault
from FastAutoAugment.train import batch_policy_decoder, augment_data
from tqdm import tqdm
from collections import Counter

def is_same_aug(augs1, augs2):
    (o11, p11, m11), (o12, p12, m12) = augs1
    (o21, p21, m21), (o22, p22, m22) = augs2
    if o11 == o21 and o12 == o22 and \
       -0.1 <= float(p11)-float(p21) <= 0.1 and -0.1 <= float(m11)-float(m21) <= 0.1 and \
       -0.1 <= float(p12)-float(p22) <= 0.1 and -0.1 <= float(m12)-float(m22) <= 0.1:
        return True
    else:
        return False

def rec_tuple(lst):
    tpl = []
    for i in range(len(lst)):
        tpl.append(tuple(lst[i]))
    tpl = tuple(tpl)
    return tpl

def main(args):
    checkpoint = torch.load(args.load_path)
    controller = Controller(n_subpolicy=5, lstm_size=100, n_group=1, gr_prob_weight=0.,\
                            img_input=True).cuda()
    controller.load_state_dict(checkpoint['ctl_state_dict'])
    _, trainloader, _, testloader =get_custom_dataloaders("cifar10", 20, "/mnt/ssd/data/", split=0.0)
    loader_iter = iter(trainloader)
    inputs, labels = loader_iter.next()
    inputs = inputs.cuda()
    pols = None
    for i in range(100):
        log_probs, entropys, sampled_policies = controller(inputs)
        batch_policies = batch_policy_decoder(sampled_policies) # (list:list:list:tuple) [batch, num_policy, n_op, 3]
        # aug_inputs, applied_policy = augment_data(inputs, batch_policies)
        if pols is None:
            pols = np.array(batch_policies)
        else:
            pols = np.concatenate((pols, np.array(batch_policies)), 1)
    # pols: (list:list:list:tuple) [batch, 100*num_policy, n_op, 3]
    cnt = 0
    bins = []
    for policy in tqdm(pols):
        bin = dict()
        in_bin = False
        for aug in policy: # aug: [n_op, 3]
            for key_aug in bin:
                if is_same_aug(aug, key_aug):
                    bin[key_aug] += 1
                    in_bin = True
                    cnt += 1
                    break
            if not in_bin:
                bin[rec_tuple(aug)] = 1
                cnt += 1
            in_bin = False
        bins.append(bin)
        print(len(bin))
        # print(bin)
        maxkey = max(bin, key=bin.get)
        print(maxkey)
        print(bin[maxkey])
    np.savez(f"{args.save_path}.npz", inputs= inputs.cpu(), pols=pols, bins=bins)
    print(cnt)

if __name__=="__main__":
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()
    main(args)
