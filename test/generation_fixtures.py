"""Explicit exact-menu oracles for control-flow tests, outside native algebra."""
import torch
from Generative import InverseMenu


def exact_menu(left,right,legal,binary=0):
    left=torch.stack(left,1) if isinstance(left,(list,tuple)) else left
    right=torch.stack(right,1) if isinstance(right,(list,tuple)) else right
    zeros=left.new_zeros(legal.shape)
    effect=zeros.clone(); effect[:,:binary]=1; effect[:,-1]=-1
    return InverseMenu(left,right,legal,legal,zeros,left.new_zeros(*legal.shape,2),effect)


def all_available(parent,left,right,available,*args,**kwargs):
    if not kwargs.get('details',False):
        return available
    return exact_menu(left,right,available,len(args[0]) if args else 0)


def suffix(tops, pending, *, choices=1):
    B,T,D=tops.shape
    return (~pending,tops.new_zeros(B,T,3),tops.new_zeros(B,1,D),
            tops.new_zeros(B,T,choices,5),tops)
