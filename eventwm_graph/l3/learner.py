"""Layer 3 learner: decomposed n-step double Q-learning over candidate sets (+ Oracle-guided warm-up losses).

Data: episodes spooled by the actors (online RL, ``source`` 0) and warm-up episodes (Oracle solutions replayed with
the Sandbox, ``source`` 1) whose decisions may carry counterfactual candidate labels (``cflabel.py``: every candidate
executed for one interval and continued by the default policy to the official end).

Value heads (``model.HEADS``) are general value functions of the greedy policy in the scalarised value
Q = w . heads.  Targets (undiscounted, episodic, n-step):
    y_h = sum_{i<n} c_h(t+i) + [no terminal within n] * head_h^target(s_{t+n}, argmax_a Q^online(s_{t+n}, a))
Losses: BCE for the probability heads, Huber for the cost heads.  Warm-up / auxiliary losses (decaying weights):
  * ``cf``     regression of every labelled candidate's heads on its counterfactual outcome -- a LOWER bound of the
               optimal value (another continuation can only be better), applied as a hinge on the scalar value
               (Q(s, c) >= Q_cf(s, c)) plus a small direct regression during the warm-up phase only;
  * ``margin`` large-margin ranking on Oracle states: Q(s, a_oracle) >= Q(s, c) + m for candidates c whose
               counterfactual outcome is a failure while the Oracle's continuation succeeds (never against other
               viable candidates -- the Oracle's choice is not assumed to be the only good one).
The online TD loss always applies to every transition, including the Oracle demonstrations (real returns).
"""
import argparse
import json
import math
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from .batch import Ring, CUM, CUM_SCALE, KMAX, GRAPH_KEYS, to_torch
from .model import ConsequenceGraphNet, ModelConfig, Weights, HEADS


def head_loss(pred, y):
    """pred, y: dict head -> [B]; returns dict of per-head mean losses."""
    out = {}
    for h in HEADS:
        if h in ('succ', 'coll'):
            out[h] = Fn.binary_cross_entropy_with_logits(pred[h], y[h].clamp(0, 1))
        else:
            out[h] = Fn.smooth_l1_loss(pred[h], y[h], beta=0.5)
    return out


class Learner:
    def __init__(self, run, device='cuda:1', capacity=1_500_000, nstep=3, batch=256, lr=3e-4, model_cfg=None,
                 weights=None, polyak=0.005, replay_ratio=32., min_size=2000, publish_every=200, seed=0,
                 head_w=None, tau_sched=(1.0, 0.25, 300_000), eps_sched=(0.05, 0.02, 300_000), cf_w=(1.0, 0.1, 100_000),
                 margin_w=(0.5, 0.05, 100_000), margin=1.0, warm_steps=0, demo_frac=0.25, demo_floor=0.03,
                 demo_decay=200_000, mc_mix=None):
        self.run = Path(run)
        self.dev = torch.device(device)
        self.ring = Ring(capacity, nstep)
        self.batch = int(batch)
        self.mcfg = model_cfg or ModelConfig()
        self.w = weights or Weights()
        self.net = ConsequenceGraphNet(self.mcfg).to(self.dev)
        self.tgt = ConsequenceGraphNet(self.mcfg).to(self.dev)
        self.tgt.load_state_dict(self.net.state_dict())
        for p in self.tgt.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.polyak = polyak
        self.replay_ratio = replay_ratio
        self.min_size = min_size
        self.publish_every = publish_every
        self.rng = np.random.default_rng(seed)
        self.head_w = head_w or dict(succ=1., coll=1., steps=0.5, path=0.5, soc=0.5)
        self.tau_sched, self.eps_sched = tau_sched, eps_sched
        self.cf_w, self.margin_w, self.margin = cf_w, margin_w, margin
        self.warm_steps = warm_steps
        self.demo_frac, self.demo_floor, self.demo_decay = demo_frac, demo_floor, demo_decay
        self.mc_mix = mc_mix              # (start, end, online decisions) weight of the Monte Carlo return in targets
        self.anchor = None                # (start, end, online decisions) actors' probability of the default candidate
        self.updates = 0
        self.version = 0
        self.loaded = set()
        self.cfg_ids = {}
        self.ep_stats = []
        self.log = open(self.run/'learner.jsonl', 'a')
        self.warm_dir, self.cf_dir, self.warm_loaded = None, None, set()
        self.warm_slots, self.warm_keys = {}, {}
        self.warm_inserted = 0
        self._last_warm = self._last_poll = self._last_cf = 0.
        self.online_slots, self.cf_applied, self.cf_online_states = {}, set(), 0
        self.cf_floor = 0.5               # counterfactual lower-bound hinge never decays below this weight
        self.cf_frac = 0.0                # share of each batch drawn from counterfactually labelled states
        self.cf_reg = 0.0                 # permanent weight of the counterfactual regression of the value heads
        self.poll_every = 5.
        self.t0 = time.time()

    # ------------------------------------------------------------------ data
    def cfg_id(self, name):
        if name not in self.cfg_ids:
            self.cfg_ids[name] = len(self.cfg_ids)
        return self.cfg_ids[name]

    def ingest_file(self, path, source=0, cf=None, require_v1b_momentum=False):
        from .actor import read_episode
        data = read_episode(path)
        rows = data['rows']
        if not rows.get('packed'):
            return 0
        if require_v1b_momentum and data.get('compiler') != 'l3_graph_v1b' and \
                data['config'].split('-')[1] in ('inertia', 'slowturn_inertia', 'heavy_brake', 'slowturn_heavy'):
            return 0                        # v1 graphs of robots with speed state differ from v1b (braking tail)
        T = len(rows['packed'])
        oracle = np.asarray(rows['actions'], np.int16) if source == 1 else None
        cfl = cfm = None
        if cf:
            cfl = np.zeros((T, KMAX, len(CUM)), np.float32)
            cfm = np.zeros((T, KMAX), bool)
            for t in range(T):
                lab = cf.get(t)
                if lab is None or lab['labels'] != rows['labels'][t]:
                    self.cf_mismatch = getattr(self, 'cf_mismatch', 0)+int(lab is not None)
                    continue                    # candidate sets differ (should not happen): no labels for this state
                for k, c in enumerate(lab['candidates'][:KMAX]):
                    cfl[t, k] = [float(c['outcome'] == 'success'), float(c['outcome'] == 'human_collision'),
                                 c['steps']/CUM_SCALE[2], c['path']/CUM_SCALE[3], c['social']/CUM_SCALE[4]]
                    cfm[t, k] = True
        idx = self.ring.add_episode(rows['packed'], rows['actions'], rows['probs'], np.stack(rows['cum']),
                                    cfg_id=self.cfg_id(data['config']), version=int(data.get('version', 0)), oracle=oracle,
                                    cf=cfl, cf_mask=cfm, source=source)
        if source == 1 and idx is not None:
            self.warm_slots[Path(path).name] = (idx, self.ring.episode[idx[0]], rows['labels'], bool(cf))
            self.warm_inserted += len(idx)
        elif source == 0 and idx is not None:
            self.online_slots[Path(path).name] = (idx, self.ring.episode[idx[0]], rows['labels'])
        m = data['metrics']
        nd = sum(int(a != b) for a, b in zip(rows['actions'], rows['default']))
        self.ep_stats.append(dict(config=data['config'], outcome=m['outcome'], success=m['success'], spl=m['spl'],
                                  psc=m['psc'], decisions=m['decisions'], version=data.get('version', 0), source=source,
                                  selector=data.get('selector'), non_default=nd))
        return T

    def ingest_warm(self):
        """Insert every warm-up episode (Oracle replay) not yet inserted, with its counterfactual labels if they exist;
        labels that arrive later are written into the ring in place (``update_cf``)."""
        if not getattr(self, 'warm_dir', None):
            return 0
        from .warmup import cf_index
        from .actor import read_episode
        cfi = cf_index(self.cf_dir) if self.cf_dir else {}
        n = 0
        for p in sorted(Path(self.warm_dir).glob('*.ep')):
            if p.name in self.warm_loaded:
                continue
            data = read_episode(p)
            m = data['metrics']
            self.warm_loaded.add(p.name)
            if m.get('diverged') or m['outcome'] != 'success':
                continue
            sp = data['spec']
            key = '%s/%s/%s/%s|%s' % (sp['dataset'], sp['split'], sp['scene'], sp['episode_id'], data['config'])
            self.warm_keys[p.name] = key
            n += self.ingest_file(p, source=1, cf=cfi.get(key) or {})
        return n

    def update_cf(self):
        """Write newly available counterfactual labels into the slots of warm episodes inserted without them."""
        if not self.cf_dir:
            return 0
        from .warmup import cf_index
        todo = [k for k, v in self.warm_slots.items() if not v[3]]
        if not todo:
            return 0
        cfi = cf_index(self.cf_dir)
        n = 0
        for name in todo:
            idx, ep_id, labels, _ = self.warm_slots[name]
            cf = cfi.get(self.warm_keys.get(name))
            if cf is None:
                continue
            with self.ring.lock:
                if not (self.ring.valid[idx].all() and (self.ring.episode[idx] == ep_id).all()):
                    continue
                for t, i in enumerate(idx):
                    lab = cf.get(t)
                    if lab is None or lab['labels'] != labels[t]:
                        continue
                    for k, c in enumerate(lab['candidates'][:KMAX]):
                        self.ring.cf[i, k] = [float(c['outcome'] == 'success'), float(c['outcome'] == 'human_collision'),
                                              c['steps']/CUM_SCALE[2], c['path']/CUM_SCALE[3], c['social']/CUM_SCALE[4]]
                        self.ring.cf_mask[i, k] = True
                    n += 1
            self.warm_slots[name] = (idx, ep_id, labels, True)
        return n

    def apply_cf_online(self):
        """Write on-policy counterfactual labels (``cfonline.py``: <run>/cf_online/<episode>.json) into the ring
        slots of their episode's transitions (when the slots still hold that episode and the candidate sets match)."""
        d = self.run/'cf_online'
        if not d.exists():
            return 0
        n = 0
        with os.scandir(d) as it:
            new = [e.name for e in it if e.name.endswith('.json') and e.name not in self.cf_applied]
        for name in new:
            ep = name[:-len('.json')]
            slot = self.online_slots.get(ep)
            if slot is None:
                if ep in self.loaded:
                    self.cf_applied.add(name)          # episode already evicted or never inserted
                continue
            try:
                rec = json.loads((d/name).read_text())
            except Exception:
                continue
            idx, ep_id, labels = slot
            with self.ring.lock:
                if self.ring.valid[idx].all() and (self.ring.episode[idx] == ep_id).all():
                    for t, st in rec['states'].items():
                        t = int(t)
                        if 'candidates' not in st or t >= len(idx) or st['labels'] != labels[t]:
                            continue
                        i = idx[t]
                        for k, c in enumerate(st['candidates'][:KMAX]):
                            # a continuation cut at the labeller's cap ('alive') is no outcome: success -1 marks it
                            # (lower-bound hinge only, no regression)
                            succ = -1. if c.get('alive') else float(c['outcome'] == 'success')
                            self.ring.cf[i, k] = [succ, float(c['outcome'] == 'human_collision'),
                                                  c['steps']/CUM_SCALE[2], c['path']/CUM_SCALE[3], c['social']/CUM_SCALE[4]]
                            self.ring.cf_mask[i, k] = True
                        n += 1
            self.cf_applied.add(name)
        self.cf_online_states += n
        return n

    def poll_spool(self, budget_s=1.0):
        """One ingestion pass on the MAIN thread (no background thread: R2's learner died of a native heap
        corruption while a watcher thread wrote into the ring and the main thread gathered batches).  New spool
        files every ``poll_every`` seconds (at most ``budget_s`` per pass); late warm-up data every 10 minutes."""
        now = time.time()
        if now-self._last_warm > 600:
            try:
                k = self.ingest_warm()
                u = self.update_cf()
                if k or u:
                    print('late warm-up: %d transitions, %d states labelled' % (k, u), flush=True)
            except Exception as exc:
                print('warm ingest failed', exc, flush=True)
            self._last_warm = now
        if now-self._last_poll < self.poll_every:
            return 0
        self._last_poll = now
        if now-self._last_cf > 60:
            try:
                self.apply_cf_online()
            except Exception as exc:
                print('cf online apply failed', exc, flush=True)
            self._last_cf = now
        n = 0
        with os.scandir(self.run/'spool') as it:
            new = sorted(e.name for e in it if e.name.endswith('.ep') and e.name not in self.loaded)
        for name in new:
            try:
                n += self.ingest_file(self.run/'spool'/name, 0)
            except Exception as exc:
                print('ingest failed', name, exc, flush=True)
            self.loaded.add(name)
            if time.time()-now > budget_s:
                break
        return n

    # ------------------------------------------------------------------ schedules
    @staticmethod
    def sched(s, step):
        a, b, n = s
        f = min(1., step/max(1, n))
        return a+(b-a)*f

    # ------------------------------------------------------------------ training
    def _heads_at(self, out, a):
        return {h: out[h].gather(1, a[:, None]).squeeze(1) for h in HEADS}

    def train_step(self):
        ring = self.ring
        B = self.batch
        has_demo = bool((ring.source[ring.valid] == 1).any())
        n_online = int((ring.source[ring.valid] == 0).sum())
        has_online = n_online > 0
        # demonstrations: a fixed share of every batch at first, decaying to a floor as online data accumulates
        frac = max(self.demo_floor, self.demo_frac*(1.-n_online/self.demo_decay))
        nd = B if not has_online else (int(B*frac) if has_demo else 0)
        idx_parts = []
        if nd:
            i1 = ring.sample(nd, self.rng, source=1)
            if i1 is not None:
                idx_parts.append(i1)
        if self.cf_frac > 0 and has_online:
            ncf = int(B*self.cf_frac)
            icf = ring.sample(ncf, self.rng, source=0, where=ring.cf_mask.any(1)) if ncf else None
            if icf is not None:
                idx_parts.append(icf)
        if B-sum(len(x) for x in idx_parts) > 0 and has_online:
            i0 = ring.sample(B-sum(len(x) for x in idx_parts), self.rng, source=0)
            if i0 is not None:
                idx_parts.append(i0)
        if not idx_parts:
            return None
        idx = np.concatenate(idx_parts)
        boot = ring.boot[idx]
        g = to_torch(ring.gather(idx), self.dev)
        g2 = to_torch(ring.gather(boot), self.dev)
        a = torch.as_tensor(ring.action[idx].astype(np.int64), device=self.dev)
        ret = torch.as_tensor(ring.ret[idx], device=self.dev)
        has_boot = torch.as_tensor(boot >= 0, device=self.dev)
        out = self.net(g)
        pred = self._heads_at(out, a)
        with torch.no_grad():
            on2 = self.net(g2)
            q2 = self.w.scalarize(on2).masked_fill(~g2['cand_mask'], -1e9)
            a2 = q2.argmax(1)
            t2 = self._heads_at(self.tgt(g2), a2)
            y = {}
            beta = self.sched(self.mc_mix, self.ring.inserted-self.warm_inserted) if self.mc_mix else 0.
            mc = torch.as_tensor(ring.mc[idx], device=self.dev) if beta > 0 else None
            for j, h in enumerate(HEADS):
                bv = torch.sigmoid(t2[h]) if h in ('succ', 'coll') else t2[h]
                y[h] = ret[:, j]+has_boot.float()*bv
                if beta > 0:            # mix in the Monte Carlo return (fast calibration; decays to pure n-step TD)
                    y[h] = (1.-beta)*y[h]+beta*mc[:, j]
        losses = head_loss(pred, y)
        loss = sum(self.head_w[h]*losses[h] for h in HEADS)
        info = {('td_'+h): float(v) for h, v in losses.items()}
        # ---------------- counterfactual lower-bound labels (warm-up states)
        cfm = torch.as_tensor(ring.cf_mask[idx], device=self.dev) & g['cand_mask']
        wcf = max(self.sched(self.cf_w, self.updates), self.cf_floor)
        if wcf > 0 and bool(cfm.any()):
            cfl = torch.as_tensor(ring.cf[idx], device=self.dev)                 # [B, K, 5]
            q = self.w.scalarize(out)
            ycf = self.w.success*cfl[..., 0].clamp(min=0)-self.w.collision*cfl[..., 1]-self.w.step*100*cfl[..., 2] \
                - self.w.path*10*cfl[..., 3]-self.w.social*50*cfl[..., 4]
            hinge = Fn.relu(ycf-q)[cfm].pow(2).mean()
            reg = 0.
            rmask = cfm & (cfl[..., 0] >= 0)                                    # complete continuations only
            if (self.updates < self.warm_steps or self.cf_reg > 0) and bool(rmask.any()):
                lab = {h: cfl[..., j] for j, h in enumerate(HEADS)}
                pr = {h: out[h][rmask] for h in HEADS}
                lb = {h: lab[h][rmask] for h in HEADS}
                reg = sum(self.head_w[h]*v for h, v in head_loss(pr, lb).items())
                if self.updates >= self.warm_steps:
                    reg = reg*self.cf_reg/max(wcf, 1e-6)      # permanent dense supervision at weight cf_reg
            loss = loss+wcf*(0.1*hinge+reg)
            info['cf_hinge'] = float(hinge)
            # ---------------- Oracle margin against counterfactual failures
            wm = self.sched(self.margin_w, self.updates)
            orc = torch.as_tensor(ring.oracle[idx].astype(np.int64), device=self.dev)
            has_o = orc >= 0
            if wm > 0 and bool(has_o.any()):
                fail = (cfl[..., 0] < 0.5) & (cfl[..., 0] >= 0) & cfm             # candidate + default fails
                qo = q.gather(1, orc.clamp(min=0)[:, None])
                viol = Fn.relu(q+self.margin-qo)*fail.float()
                viol = viol*has_o[:, None].float()
                viol = viol.scatter(1, orc.clamp(min=0)[:, None], 0.)
                ml = viol.sum()/max(1., float(fail[has_o].sum()))
                loss = loss+wm*ml
                info['margin'] = float(ml)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.)
        self.opt.step()
        with torch.no_grad():
            for p, pt in zip(self.net.parameters(), self.tgt.parameters()):
                pt.mul_(1-self.polyak).add_(p, alpha=self.polyak)
        self.updates += 1
        info.update(loss=float(loss), grad_norm=float(gn))
        return info

    def publish(self):
        self.version += 1
        sd = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        ck = dict(state_dict=sd, model_config=asdict(self.mcfg), weights=asdict(self.w), version=self.version,
                  updates=self.updates, tau=self.sched(self.tau_sched, self.ring.inserted-self.warm_inserted),
                  epsilon=self.sched(self.eps_sched, self.ring.inserted-self.warm_inserted), time=time.time(),
                  anchor=self.sched(self.anchor, self.ring.inserted-self.warm_inserted) if self.anchor else 0.,
                  cfg_ids=self.cfg_ids)
        tmp = self.run/'.policy.pt.tmp'
        torch.save(ck, tmp)
        os.replace(tmp, self.run/'policy.pt')
        if self.version % 25 == 0:
            (self.run/'checkpoints').mkdir(exist_ok=True)
            torch.save(ck, self.run/'checkpoints'/('policy_v%05d.pt' % self.version))

    def save_full(self, name='learner_state.pt'):
        tmp = self.run/('.%s.tmp' % name)
        torch.save(dict(net=self.net.state_dict(), tgt=self.tgt.state_dict(), opt=self.opt.state_dict(),
                        updates=self.updates, version=self.version, model_config=asdict(self.mcfg),
                        weights=asdict(self.w)), tmp)
        os.replace(tmp, self.run/name)

    def load_init(self, path):
        ck = torch.load(path, map_location=self.dev)
        sd = ck.get('net') or ck.get('state_dict')
        self.net.load_state_dict(sd)
        self.tgt.load_state_dict(ck.get('tgt') or sd)
        if 'opt' in ck:                       # a full learner state (restart): continue counters and optimiser
            self.opt.load_state_dict(ck['opt'])
            self.updates = int(ck.get('updates', 0))
            self.version = int(ck.get('version', 0))

    def loop(self, max_updates=10**12, max_seconds=1e12, pretrain_updates=0):
        last_log = last_save = time.time()
        acc = {}
        try:
            if pretrain_updates:
                for _ in range(pretrain_updates):
                    info = self.train_step()
                    for k, v in (info or {}).items():
                        acc.setdefault(k, []).append(v)
                    if self.updates % 500 == 0:
                        self._log(acc, phase='pretrain'); acc = {}
                self.publish()
            self.publish()
            base = self.updates                      # replay ratio counts online updates against online data only
            while self.updates < max_updates and time.time()-self.t0 < max_seconds:
                self.poll_spool()
                online = int((self.ring.source[self.ring.valid] == 0).sum())
                online_in = self.ring.inserted-self.warm_inserted
                if online < self.min_size or (self.updates-base)*self.batch > self.replay_ratio*max(online_in, 1):
                    time.sleep(0.2)
                else:
                    info = self.train_step()
                    for k, v in (info or {}).items():
                        acc.setdefault(k, []).append(v)
                    if self.updates % self.publish_every == 0:
                        self.publish()
                if time.time()-last_log > 60:
                    self._log(acc); acc = {}
                    last_log = time.time()
                if time.time()-last_save > 600:
                    self.save_full()                      # the fleet restarts a dead learner from this state
                    last_save = time.time()
        finally:
            self.save_full()
            self.publish()

    def _log(self, acc, phase='rl'):
        rec = dict(t=round(time.time()-self.t0, 1), phase=phase, updates=self.updates, version=self.version,
                   ring=int(self.ring.valid.sum()), inserted=int(self.ring.inserted), episodes=len(self.ep_stats),
                   cf_online_states=self.cf_online_states,
                   **{k: round(float(np.mean(v)), 5) for k, v in acc.items()})
        recent = [e for e in self.ep_stats[-400:] if e['source'] == 0]
        by = {}
        for e in recent:
            by.setdefault(e['config'], []).append(e)
        rec['recent'] = {c: dict(n=len(v), sr=round(float(np.mean([x['success'] for x in v])), 3),
                                 spl=round(float(np.mean([x['spl'] for x in v])), 3),
                                 coll=round(float(np.mean([x['outcome'] == 'human_collision' for x in v])), 3),
                                 nondef=round(float(sum(x['non_default'] for x in v)/max(1, sum(x['decisions'] for x in v))), 3))
                         for c, v in sorted(by.items())}
        self.log.write(json.dumps(rec)+'\n'); self.log.flush()
        print(json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--init', default=None)
    ap.add_argument('--warm', default=None, help='warm-up spool directory (Oracle replays) with cf labels directory')
    ap.add_argument('--cf', default=None, help='counterfactual label directory (cflabel.py output)')
    ap.add_argument('--pretrain', type=int, default=0)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--replay-ratio', type=float, default=32.)
    ap.add_argument('--nstep', type=int, default=3)
    ap.add_argument('--min-size', type=int, default=2000)
    ap.add_argument('--max-seconds', type=float, default=1e12)
    ap.add_argument('--model', default='{}')
    ap.add_argument('--weights', default='{}')
    ap.add_argument('--tau', default='1.0,0.25,300000')
    ap.add_argument('--eps', default='0.05,0.02,300000')
    ap.add_argument('--mc-mix', default=None, help='start,end,decisions: Monte Carlo share of the value targets')
    ap.add_argument('--seed-spool', default=None, help='ingest the episodes of another run first (off-policy reuse)')
    ap.add_argument('--offline-updates', type=int, default=0,
                    help='offline mode: train this many updates on the seeded + warm data (no actors), publish, exit')
    ap.add_argument('--seed-limit', type=int, default=0, help='ingest at most this many seed episodes (0 = all)')
    ap.add_argument('--seed-configs', default=None, help='only seed episodes of these configurations (comma list)')
    ap.add_argument('--anchor', default=None, help='start,end,decisions: default-anchored exploration probability')
    ap.add_argument('--cf-frac', type=float, default=0., help='share of each batch from counterfactually labelled states')
    ap.add_argument('--v1b', action='store_true', help='seed only v1b graphs for robots with speed state')
    ap.add_argument('--cf-floor', type=float, default=0.5, help='minimum weight of the counterfactual hinge')
    ap.add_argument('--cf-reg', type=float, default=0., help='permanent counterfactual regression weight on the value heads')
    a = ap.parse_args()
    run = Path(a.run); run.mkdir(parents=True, exist_ok=True)
    import signal
    import sys
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))      # run the loop's finally (state save, final publish)
    mc = json.loads(a.model)
    if 'drop' in mc:
        mc['drop'] = tuple(mc['drop'])
    tau = tuple(float(x) for x in a.tau.split(','))
    eps = tuple(float(x) for x in a.eps.split(','))
    L = Learner(run, a.device, nstep=a.nstep, batch=a.batch, lr=a.lr, model_cfg=ModelConfig(**mc),
                weights=Weights(**json.loads(a.weights)), replay_ratio=a.replay_ratio, min_size=a.min_size,
                tau_sched=(tau[0], tau[1], int(tau[2])), eps_sched=(eps[0], eps[1], int(eps[2])),
                warm_steps=a.pretrain,
                mc_mix=tuple(float(x) for x in a.mc_mix.split(',')) if a.mc_mix else None)
    (run/'learner_args.json').write_text(json.dumps(vars(a)))
    if a.anchor:
        L.anchor = tuple(float(x) for x in a.anchor.split(','))
    L.cf_frac, L.cf_floor, L.cf_reg = a.cf_frac, a.cf_floor, a.cf_reg
    if a.init:
        L.load_init(a.init)
    if a.seed_spool:
        n = 0
        files = sorted((p for d in a.seed_spool.split(',') for p in Path(d).glob('*.ep')), key=lambda p: p.stat().st_mtime)
        if a.seed_limit:
            files = files[:a.seed_limit]
        keep = set(a.seed_configs.split(',')) if a.seed_configs else None
        for p in files:
            if keep is not None and p.name.split('__')[1] not in keep:
                continue
            try:
                n += L.ingest_file(p, 0, require_v1b_momentum=a.v1b)
            except Exception as exc:
                print('seed ingest failed', p, exc, flush=True)
        print('seeded online transitions', n, 'from', a.seed_spool, flush=True)
    if a.warm:
        L.warm_dir, L.cf_dir = a.warm, a.cf
        n = L.ingest_warm()
        print('warm-up transitions', n, 'cf mismatches', getattr(L, 'cf_mismatch', 0), flush=True)
    if a.offline_updates:
        acc = {}
        for i in range(a.pretrain+a.offline_updates):
            info = L.train_step()
            for k, v in (info or {}).items():
                acc.setdefault(k, []).append(v)
            if (i+1) % 1000 == 0:
                L._log(acc, phase='offline'); acc = {}
            if (i+1) % 5000 == 0:
                L.publish()
        L.publish()
        L.save_full()
        return
    L.loop(max_seconds=a.max_seconds, pretrain_updates=a.pretrain)


if __name__ == '__main__':
    main()
