"""Shared Layer 3 selector: a consequence-graph network with decomposed value heads.

Input: a padded batch of compiled consequence graphs (``batch.py``).  Message passing follows the graph's relations:

  1. human-future -> trajectory at the same time (attention over tracked people, edge features = pair statistics)
  2. temporal mixing along each imagined trajectory (self-attention over the P waypoints)
  3. trajectory -> candidate (attention pooling over waypoints)
  4. human-future -> human (pooling over time), humans -> ego (attention)
  5. candidate set reasoning: a transformer layer over {ego, candidates} (comparisons between alternatives)

Readout per candidate: decomposed action values (general value functions under the greedy policy):
  succ  P(eventual official success | choose c, then the policy)          (logit)
  coll  P(eventual human collision | ...)                                  (logit)
  steps expected remaining native steps / 100
  path  expected remaining path length / 10 m
  soc   expected remaining social steps (any human < 1 m) / 50
The scalar action value is a fixed linear scalarisation (``Weights``); the policy is greedy / Boltzmann in it.

``drop`` (feature-group ablations) zeroes compiler feature groups at the input, and ``flat`` removes the
waypoint / human-future / interaction nodes (a set model over candidate summaries).
"""
from dataclasses import dataclass, asdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import compiler as C

HEADS = ('succ', 'coll', 'steps', 'path', 'soc')
SCALE = dict(steps=100., path=10., soc=50.)


@dataclass
class Weights:
    success: float = 10.
    collision: float = 5.
    step: float = 0.002
    path: float = 0.3
    social: float = 0.02

    def scalarize(self, heads):
        """heads: dict of tensors [..., K] in network units -> scalar Q [..., K] (reward units)."""
        return (self.success*torch.sigmoid(heads['succ'])-self.collision*torch.sigmoid(heads['coll'])
                - self.step*SCALE['steps']*heads['steps']-self.path*SCALE['path']*heads['path']
                - self.social*SCALE['soc']*heads['soc'])

    def reward(self, success, collision, steps, path, social):
        return self.success*success-self.collision*collision-self.step*steps-self.path*path-self.social*social


@dataclass
class ModelConfig:
    hidden: int = 128
    edge: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0
    flat: bool = False
    drop: tuple = ()              # compiler feature groups zeroed at the input (ablations)


def mlp(i, h, o, n=2, act=nn.GELU):
    layers = []
    d = i
    for _ in range(n-1):
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def masked_softmax(logits, mask, dim):
    logits = logits.masked_fill(~mask, -1e9)
    w = torch.softmax(logits, dim=dim)
    return w*mask.to(w.dtype)


class Block(nn.Module):
    """One round of heterogeneous message passing on the dense consequence graph."""

    def __init__(self, cfg):
        super().__init__()
        H, E, nh = cfg.hidden, cfg.edge, cfg.heads
        self.cfg = cfg
        self.flat = cfg.flat
        if not cfg.flat:
            # 1. hfut(n, p) -> traj(k, p) attention with pair features
            self.q_int = nn.Linear(H, H)
            self.k_int = nn.Linear(H+E, H)
            self.v_int = nn.Linear(H+E, H)
            self.null_k = nn.Parameter(torch.zeros(H))
            self.null_v = nn.Parameter(torch.zeros(H))
            self.u_traj = mlp(2*H, H, H)
            self.n_traj = nn.LayerNorm(H)
            # 2. temporal self-attention along P
            self.temporal = nn.MultiheadAttention(H, nh, batch_first=True)
            self.n_temp = nn.LayerNorm(H)
            # 3. traj -> cand pooling
            self.q_pool = nn.Linear(H, H)
            self.k_pool = nn.Linear(H, H)
            self.u_cand = mlp(3*H, H, H)
            self.n_cand = nn.LayerNorm(H)
            # 4. hfut -> hum -> ego
            self.u_hum = mlp(2*H, H, H)
            self.n_hum = nn.LayerNorm(H)
            self.q_ego = nn.Linear(H, H)
            self.k_hum = nn.Linear(H, H)
            self.u_ego = mlp(2*H, H, H)
            self.n_ego = nn.LayerNorm(H)
        # 5. candidate set transformer (ego token first)
        self.setattn = nn.MultiheadAttention(H, nh, batch_first=True)
        self.n_set = nn.LayerNorm(H)
        self.ff = mlp(H, 2*H, H)
        self.n_ff = nn.LayerNorm(H)

    def forward(self, x, m):
        H = self.cfg.hidden
        ego, cand = x['ego'], x['cand']                       # [B, H], [B, K, H]
        cmask = m['cand']                                     # [B, K]
        if not self.flat:
            traj, hum, hfut, pair = x['traj'], x['hum'], x['hfut'], x['pair']   # [B,K,P,H], [B,N,H], [B,N,P,H], [B,K,P,N,E]
            hmask = m['hum']                                  # [B, N]
            B, K, P, _ = traj.shape
            N = hum.shape[1]
            # 1. interaction: traj(k,p) attends over humans n at the same p (+ a null slot)
            q = self.q_int(traj)                              # [B,K,P,H]
            if N:
                kv_in = torch.cat([hfut.unsqueeze(1).expand(B, K, N, P, H).transpose(2, 3), pair], -1)   # [B,K,P,N,H+E]
                kk = self.k_int(kv_in); vv = self.v_int(kv_in)
                logit = (q.unsqueeze(3)*kk).sum(-1)/math.sqrt(H)                                      # [B,K,P,N]
                null = (q*self.null_k).sum(-1, keepdim=True)/math.sqrt(H)                             # [B,K,P,1]
                logits = torch.cat([null, logit], -1)
                mk = torch.cat([torch.ones_like(cmask[:, :1]).unsqueeze(1).unsqueeze(1).expand(B, K, P, 1),
                                hmask.unsqueeze(1).unsqueeze(1).expand(B, K, P, N)], -1)
                w = masked_softmax(logits, mk, -1)
                msg = w[..., :1]*self.null_v+(w[..., 1:].unsqueeze(-1)*vv).sum(-2)
            else:
                msg = self.null_v.expand(B, K, P, H)
            traj = self.n_traj(traj+self.u_traj(torch.cat([traj, msg], -1)))
            # 2. temporal mixing within each candidate (+ the candidate's own embedding as context)
            t2 = (traj+cand.unsqueeze(2)).reshape(B*K, P, H)
            a, _ = self.temporal(t2, t2, t2, need_weights=False)
            traj = self.n_temp(traj+a.reshape(B, K, P, H))
            # 3. pooling waypoints into the candidate
            pl = (self.q_pool(cand).unsqueeze(2)*self.k_pool(traj)).sum(-1)/math.sqrt(H)          # [B,K,P]
            pw = torch.softmax(pl, -1)
            pooled = (pw.unsqueeze(-1)*traj).sum(2)
            cand = self.n_cand(cand+self.u_cand(torch.cat([cand, pooled, traj.max(2).values], -1)))
            # 4. humans <- their futures; ego <- humans
            if N:
                hum = self.n_hum(hum+self.u_hum(torch.cat([hum, hfut.mean(2)], -1)))
                el = (self.q_ego(ego).unsqueeze(1)*self.k_hum(hum)).sum(-1)/math.sqrt(H)          # [B,N]
                any_h = hmask.any(-1, keepdim=True)
                ew = masked_softmax(el, hmask, -1)
                hm = (ew.unsqueeze(-1)*hum).sum(1)*any_h.to(hum.dtype)
            else:
                hm = torch.zeros_like(ego)
            ego = self.n_ego(ego+self.u_ego(torch.cat([ego, hm], -1)))
            x = dict(x, traj=traj, hum=hum, hfut=hfut)
        # 5. candidate set reasoning
        tokens = torch.cat([ego.unsqueeze(1), cand], 1)                                           # [B,1+K,H]
        tmask = torch.cat([torch.ones_like(cmask[:, :1]), cmask], 1)
        a, _ = self.setattn(tokens, tokens, tokens, key_padding_mask=~tmask, need_weights=False)
        tokens = self.n_set(tokens+a)
        tokens = self.n_ff(tokens+self.ff(tokens))
        x = dict(x, ego=tokens[:, 0], cand=tokens[:, 1:])
        return x


class ConsequenceGraphNet(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        H, E = cfg.hidden, cfg.edge
        self.enc = nn.ModuleDict(dict(ego=mlp(len(C.EGO), H, H), cand=mlp(len(C.CAND), H, H), traj=mlp(len(C.TRAJ), H, H),
                                      hum=mlp(len(C.HUM), H, H), hfut=mlp(len(C.HFUT), H, H), pair=mlp(len(C.PAIR), E, E)))
        self.time_emb = nn.Parameter(torch.zeros(C.P, H))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.head = mlp(2*H, H, len(HEADS))
        # input masks for feature-group ablations
        for name, cols in (('ego', C.EGO), ('cand', C.CAND), ('traj', C.TRAJ)):
            keep = torch.ones(len(cols))
            for g in cfg.drop:
                for c in C.GROUPS[g].get(name, []):
                    keep[cols.index(c)] = 0.
            self.register_buffer('keep_'+name, keep)

    def forward(self, b):
        """b: dict of padded tensors (batch.py).  Returns dict head -> [B, K]."""
        H = self.cfg.hidden
        x = dict(ego=self.enc['ego'](b['ego']*self.keep_ego), cand=self.enc['cand'](b['cand']*self.keep_cand))
        m = dict(cand=b['cand_mask'], hum=b['hum_mask'])
        if not self.cfg.flat:
            x['traj'] = self.enc['traj'](b['traj']*self.keep_traj)+self.time_emb
            x['hum'] = self.enc['hum'](b['hum'])
            x['hfut'] = self.enc['hfut'](b['hfut'])+self.time_emb
            x['pair'] = self.enc['pair'](b['pair'])
            if 'human' in self.cfg.drop:
                x['hum'] = torch.zeros_like(x['hum']); x['hfut'] = torch.zeros_like(x['hfut'])
                x['pair'] = torch.zeros_like(x['pair'])
        for blk in self.blocks:
            x = blk(x, m)
        K = x['cand'].shape[1]
        h = self.head(torch.cat([x['cand'], x['ego'].unsqueeze(1).expand(-1, K, -1)], -1))     # [B,K,5]
        return {name: h[..., i] for i, name in enumerate(HEADS)}


def masked_q(model_out, weights, mask):
    q = weights.scalarize(model_out)
    return q.masked_fill(~mask, -1e9)
