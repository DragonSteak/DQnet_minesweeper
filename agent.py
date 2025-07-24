import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from model import QNet

class DQNAgent:
    def __init__(self):
        self.net = QNet()
        self.tgt = QNet()
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.buffer = deque(maxlen=20000)
        self.bs = 64
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995
        self.steps = 0

    def select(self, s):
        valid = s.reshape(-1) == -0.125
        if random.random() < self.eps:
            return random.choice(np.where(valid)[0])
        with torch.no_grad():
            q = self.net(torch.tensor(s.transpose(2, 0, 1)).unsqueeze(0))
            q = q.squeeze().numpy()
            q[~valid] = -np.inf
            return int(np.argmax(q))

    def store(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def update(self):
        if len(self.buffer) < self.bs: return
        batch = random.sample(self.buffer, self.bs)
        s, a, r, s2, d = zip(*batch)

        s = torch.tensor(np.array(s).transpose(0, 3, 1, 2), dtype=torch.float32)
        s2 = torch.tensor(np.array(s2).transpose(0, 3, 1, 2), dtype=torch.float32)
        a = torch.tensor(a)
        r = torch.tensor(r, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)

        qvals = self.net(s).gather(1, a.unsqueeze(1)).squeeze()
        next_q = self.tgt(s2).max(1)[0]
        target = r + self.gamma * next_q * (1 - d)

        loss = F.mse_loss(qvals, target.detach())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.steps += 1
        if self.steps % 200 == 0:
            self.tgt.load_state_dict(self.net.state_dict())
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
