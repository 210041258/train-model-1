#!/usr/bin/env python3
"""
[init] flows=14  intersections=12  state_dim=50  envs=210  device=cpu
[init] intersections: [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3)]
[init] grad_steps/iter=26  batch=256  buffer=200000



main.py -- DQN traffic-signal training using ONLY flow_3_4.json.

No roadnet.json and no CityFlow engine.  The simulator is built by
reconstructing the road/intersection structure from the road naming
convention used inside the flow file:

    road_<col>_<row>_<dir>      dir: 0=E, 1=S, 2=W, 3=N

An intersection at grid point (x, y) is treated as a 4-way signalised
intersection iff all four approach roads exist:

    road_{x-1}_{y}_0   (from West)
    road_{x+1}_{y}_2   (from East)
    road_{x}_{y-1}_1   (from North)
    road_{x}_{y+1}_3   (from South)

For flow_3_4.json this yields a 4x3 grid of 12 intersections.

Trains a Double-DQN for --total-steps decisions (default 1,000,000)
across --num-envs parallel simulators (default 210).

Examples
--------
    python main.py                         # 1M steps / 210 envs
    python main.py --dry-run               # 5 envs, 2k steps, 30 s
    python main.py --num-envs 32 --total-steps 500000
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# --------------------------------------------------------------------------- #
# Flow-file parsing
# --------------------------------------------------------------------------- #
def parse_road(road_id: str):
    """'road_3_1_0' -> (3, 1, 0)."""
    p = road_id.split("_")
    return int(p[1]), int(p[2]), int(p[3])


def infer_intersections(flows) -> dict:
    """Return {(x, y): {'N': road, 'S': road, 'E': road, 'W': road}}."""
    roads = set()
    for flow in flows:
        roads.update(flow["route"])

    candidates = set()
    for r in roads:
        x, y, d = parse_road(r)
        if   d == 0: candidates.add((x + 1, y))
        elif d == 2: candidates.add((x - 1, y))
        elif d == 1: candidates.add((x, y + 1))
        elif d == 3: candidates.add((x, y - 1))

    out = {}
    for (x, y) in candidates:
        needs = {
            "W": f"road_{x - 1}_{y}_0",
            "E": f"road_{x + 1}_{y}_2",
            "N": f"road_{x}_{y - 1}_1",
            "S": f"road_{x}_{y + 1}_3",
        }
        if all(v in roads for v in needs.values()):
            out[(x, y)] = needs
    return out


# direction index:  0=N, 1=S, 2=E, 3=W
def build_inflow(flows, intersections):
    inter_list = sorted(intersections.keys())
    idx_of = {it: i for i, it in enumerate(inter_list)}
    rate = np.zeros((len(inter_list), 4), dtype=np.float32)

    for flow in flows:
        r = 1.0 / float(flow["interval"])
        for road in flow["route"]:
            x, y, d = parse_road(road)
            if   d == 0: ix, iy, adir = x + 1, y, 3   # enters from West
            elif d == 2: ix, iy, adir = x - 1, y, 2   # enters from East
            elif d == 1: ix, iy, adir = x, y + 1, 0   # enters from North
            elif d == 3: ix, iy, adir = x, y - 1, 1   # enters from South
            else:        continue
            if (ix, iy) in idx_of:
                rate[idx_of[(ix, iy)], adir] += r
    return inter_list, rate


# --------------------------------------------------------------------------- #
# Vectorised flow simulator
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    dt: float = 1.0
    discharge_rate: float = 2.0     # veh/s per approach when green
    action_steps: int = 5           # sim steps per decision
    episode_decisions: int = 100    # decisions per episode


class VecFlowSim:
    """N independent queue simulators driven by a single numpy state."""

    def __init__(self, flows, num_envs: int, seed: int, cfg: SimConfig):
        self.cfg = cfg
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)

        inters = infer_intersections(flows)
        self.inter_list, self.inflow_rate = build_inflow(flows, inters)
        self.n_inter = len(self.inter_list)
        self.n_dir = 4
        self.action_dim = 2
        self.state_dim = self.n_inter * self.n_dir + 2

        self.queues = np.zeros((num_envs, self.n_inter, self.n_dir), dtype=np.float32)
        self.phases = np.zeros(num_envs, dtype=np.int64)
        self.decision_steps = np.zeros(num_envs, dtype=np.int64)

    # ---- API ----
    def reset(self, env_ids=None):
        if env_ids is None:
            self.queues[:] = 0.0
            self.phases[:] = 0
            self.decision_steps[:] = 0
        else:
            self.queues[env_ids] = 0.0
            self.phases[env_ids] = 0
            self.decision_steps[env_ids] = 0
        return self._state()

    def _state(self) -> np.ndarray:
        q = self.queues.reshape(self.num_envs, -1)
        ph = np.eye(2, dtype=np.float32)[self.phases]
        return np.concatenate([q, ph], axis=1).astype(np.float32)

    def step(self, actions: np.ndarray):
        """actions: (num_envs,) in {0,1}. Returns (s, r, done)."""
        actions = np.asarray(actions, dtype=np.int64)
        self.phases = actions.copy()

        total_q = np.zeros(self.num_envs, dtype=np.float32)
        dis = self.cfg.discharge_rate * self.cfg.dt
        ns = (actions == 0)[:, None]     # NS green -> N & S discharge
        ew = (actions == 1)[:, None]     # EW green -> E & W discharge

        for _ in range(self.cfg.action_steps):
            arrivals = self.rng.poisson(
                self.inflow_rate[None] * self.cfg.dt,
                size=(self.num_envs, self.n_inter, self.n_dir),
            ).astype(np.float32)

            outflow = np.zeros_like(self.queues)
            outflow[:, :, 0] = ns * dis     # N approach
            outflow[:, :, 1] = ns * dis     # S approach
            outflow[:, :, 2] = ew * dis     # E approach
            outflow[:, :, 3] = ew * dis     # W approach
            outflow = np.minimum(outflow, self.queues + arrivals)

            self.queues = np.maximum(self.queues + arrivals - outflow, 0.0)
            total_q += self.queues.sum(axis=(1, 2))

        self.decision_steps += 1
        denom = self.cfg.action_steps * self.n_inter * self.n_dir
        reward = -total_q / denom

        dones = self.decision_steps >= self.cfg.episode_decisions
        if dones.any():
            self.reset(np.where(dones)[0])

        return self._state(), reward, dones


# --------------------------------------------------------------------------- #
# DQN
# --------------------------------------------------------------------------- #
class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.s  = np.zeros((capacity, state_dim), dtype=np.float32)
        self.ns = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a  = np.zeros(capacity, dtype=np.int64)
        self.r  = np.zeros(capacity, dtype=np.float32)
        self.d  = np.zeros(capacity, dtype=np.float32)
        self.ptr = self.size = self.capacity = capacity
        self.ptr = 0
        self.size = 0

    def push_batch(self, s, a, r, ns, d):
        n = len(a)
        end = self.ptr + n
        if end <= self.capacity:
            self.s[self.ptr:end]  = s
            self.a[self.ptr:end]  = a
            self.r[self.ptr:end]  = r
            self.ns[self.ptr:end] = ns
            self.d[self.ptr:end]  = d
        else:
            wrap = end - self.capacity
            first = n - wrap
            self.s[self.ptr:]  = s[:first];   self.s[:wrap]  = s[first:]
            self.a[self.ptr:]  = a[:first];   self.a[:wrap]  = a[first:]
            self.r[self.ptr:]  = r[:first];   self.r[:wrap]  = r[first:]
            self.ns[self.ptr:] = ns[:first];  self.ns[:wrap] = ns[first:]
            self.d[self.ptr:]  = d[:first];   self.d[:wrap]  = d[first:]
        self.ptr = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch: int):
        idx = np.random.randint(0, self.size, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.ns[idx], self.d[idx]


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    with open(args.flow_file) as f:
        flows = json.load(f)

    sim_cfg = SimConfig(dt=args.dt,
                        discharge_rate=args.discharge_rate,
                        action_steps=args.action_steps,
                        episode_decisions=args.episode_decisions)
    sim = VecFlowSim(flows, num_envs=args.num_envs, seed=args.seed, cfg=sim_cfg)

    n_inter = sim.n_inter
    print(f"[init] flows={len(flows)}  intersections={n_inter}  "
          f"state_dim={sim.state_dim}  envs={args.num_envs}  device={device}")
    print(f"[init] intersections: {sim.inter_list}")

    q_net   = QNet(sim.state_dim, sim.action_dim, args.hidden).to(device)
    tgt_net = QNet(sim.state_dim, sim.action_dim, args.hidden).to(device)
    tgt_net.load_state_dict(q_net.state_dict())
    tgt_net.eval()
    optimiser = optim.Adam(q_net.parameters(), lr=args.lr)

    buffer = ReplayBuffer(args.buffer_size, sim.state_dim)

    grad_steps = args.grad_steps if args.grad_steps > 0 else max(1, args.num_envs // 8)
    print(f"[init] grad_steps/iter={grad_steps}  "
          f"batch={args.batch_size}  buffer={args.buffer_size}")

    states = sim.reset()
    total = 0
    it = 0
    ep_rewards = np.zeros(args.num_envs, dtype=np.float32)
    reward_window = deque(maxlen=50)
    t0 = 0.0
    import time
    t0 = time.time()

    while total < args.total_steps:
        it += 1
        eps = max(args.eps_end,
                  args.eps_start - (args.eps_start - args.eps_end) * total / args.eps_decay_steps)

        if random.random() < eps:
            actions = np.random.randint(0, sim.action_dim, size=args.num_envs)
        else:
            with torch.no_grad():
                q = q_net(torch.as_tensor(states, device=device))
                actions = q.argmax(1).cpu().numpy()

        next_states, rewards, dones = sim.step(actions)
        total += args.num_envs
        buffer.push_batch(states, actions, rewards, next_states, dones.astype(np.float32))

        ep_rewards += rewards
        if dones.any():
            for i in np.where(dones)[0]:
                reward_window.append(float(ep_rewards[i]))
                ep_rewards[i] = 0.0

        states = next_states

        if buffer.size >= args.train_start:
            for _ in range(grad_steps):
                s, a, r, ns, d = buffer.sample(args.batch_size)
                s  = torch.as_tensor(s,  device=device)
                a  = torch.as_tensor(a,  device=device)
                r  = torch.as_tensor(r,  device=device)
                ns = torch.as_tensor(ns, device=device)
                d  = torch.as_tensor(d,  device=device)

                qv = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    na = q_net(ns).argmax(1)
                    nq = tgt_net(ns).gather(1, na.unsqueeze(1)).squeeze(1)
                    tq = r + args.gamma * nq * (1.0 - d)
                loss = F.smooth_l1_loss(qv, tq)
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimiser.step()

        if it % args.target_update_steps == 0:
            tgt_net.load_state_dict(q_net.state_dict())

        if it % args.log_every == 0:
            avg = float(np.mean(reward_window)) if reward_window else float("nan")
            sps = total / max(time.time() - t0, 1e-6)
            print(f"[{total:>8d}/{args.total_steps}] it={it:<6d} eps={eps:.3f} "
                  f"buf={buffer.size:>6d} avg_ep_R={avg:8.3f} sps={sps:7.1f}")

    torch.save({"q_net": q_net.state_dict(), "args": vars(args)}, args.ckpt)
    print(f"[done] saved {args.ckpt}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # scenario
    p.add_argument("--flow-file", default="/content/flow_3_4.json")
    p.add_argument("--action-steps", type=int, default=5)
    p.add_argument("--episode-decisions", type=int, default=100)
    p.add_argument("--discharge-rate", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=1.0)
    # training
    p.add_argument("--num-envs", type=int, default=210)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--train-start", type=int, default=10_000)
    p.add_argument("--grad-steps", type=int, default=0,
                   help="gradient steps per iteration; 0 = auto")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=100_000)
    p.add_argument("--target-update-steps", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt", default="final.pt")
    # quick test
    p.add_argument("--dry-run", action="store_true")

    # Use parse_known_args to avoid exceptions with notebook launcher arguments
    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()
    if args.dry_run:
        args.num_envs = min(5, args.num_envs)
        args.total_steps = min(2_000, args.total_steps)
        args.train_start = 200
        args.batch_size = 32
        args.log_every = 5
        args.grad_steps = 2
    try:
        train(args)
    except KeyboardInterrupt:
        print("\n[interrupted]")


if __name__ == "__main__":
    main()
