"""Causal Transformer core and state-inferred contact residuals.

No geometry, gravity equation, clock, future state, or true contact is used
in inference. Normalizations and optional affine initializations are fitted
on training transitions. History is left-padded by repeating the first state.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from nedm.core.training.model_transformer import ContinuousTransformer, TransformerConfig


class TemporalContactNRD(nn.Module):
    def __init__(self, config, normalization):
        super().__init__()
        self.config = dict(config)
        self.dt = float(config['dt_s'])
        self.context = int(config.get('context', 1))
        self.contact_enabled = bool(config.get('contact_enabled', True))
        self.modes = int(config.get('contact_modes', 2))
        self.bounce_input_dim = int(config.get('bounce_input_dim', 5))
        dtype = torch.float64 if config.get('dtype', 'float64') == 'float64' else torch.float32
        for key in ('mean', 'std', 'delta_mean'):
            self.register_buffer('state_'+key if key != 'delta_mean' else key,
                                 torch.tensor(normalization[key], dtype=dtype))
        self.register_buffer('delta_scale', torch.tensor(config.get('delta_scale', [.05,.05,2.,4.,10.]), dtype=dtype))
        width = int(config.get('embedding', 128))
        self.backbone = ContinuousTransformer(TransformerConfig(5, self.context,
            config.get('layers', 4), config.get('heads', 4), width, 0., True))
        self.affine_preserving = bool(config.get('affine_preserving', False))
        if self.affine_preserving:
            self.backbone.final_norm = nn.Identity()
            nn.init.zeros_(self.backbone.position_embedding.weight)
            if config.get('identity_init',True):
                for block in self.backbone.blocks:
                    nn.init.zeros_(block.attn.c_proj.weight)
                    nn.init.zeros_(block.attn.c_proj.bias)
                    nn.init.zeros_(block.mlp.c_proj.weight)
                    nn.init.zeros_(block.mlp.c_proj.bias)
            self.core_head = nn.Linear(width, 5)
        else:
            self.core_head = nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,5))
        last = self.core_head if self.affine_preserving else self.core_head[-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        self.gate_input_dim = int(config.get('gate_input_dim', 5))
        hidden = int(config.get('hidden', 128))
        gate_hidden=int(config.get('gate_hidden',hidden))
        gate_activation=nn.ReLU if config.get('gate_activation','tanh')=='relu' else nn.Tanh
        self.contact_encoder = nn.Sequential(nn.Linear(self.gate_input_dim,gate_hidden),gate_activation(),
            nn.Linear(gate_hidden,gate_hidden),gate_activation(),nn.Linear(gate_hidden,32),gate_activation())
        self.gate_modes = int(config.get('gate_modes',self.modes))
        self.contact_head = nn.Linear(32,self.gate_modes)
        self.contact_linear = nn.Linear(self.gate_input_dim,self.gate_modes)
        nn.init.zeros_(self.contact_linear.weight); nn.init.zeros_(self.contact_linear.bias)
        activation = nn.Tanh if config.get('bounce_activation','tanh')=='tanh' else nn.GELU
        def response_net():
            layers=[nn.Linear(self.bounce_input_dim,hidden),activation()]
            for _ in range(int(config.get('bounce_layers',2))-1):layers.extend((nn.Linear(hidden,hidden),activation()))
            return nn.Sequential(*layers,nn.Linear(hidden,5))
        self.bounce = nn.ModuleList([response_net() for _ in range(self.modes)])
        self.bounce_linear = nn.ModuleList([nn.Linear(self.bounce_input_dim,5) for _ in range(self.modes)])
        self.register_buffer('bounce_mean',torch.zeros(self.modes,self.bounce_input_dim,dtype=dtype))
        self.register_buffer('bounce_std',torch.ones(self.modes,self.bounce_input_dim,dtype=dtype))
        self.register_buffer('response_mean',torch.zeros(self.modes,5,dtype=dtype))
        self.register_buffer('response_std',torch.ones(self.modes,5,dtype=dtype))
        for net in self.bounce:
            nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
        self.to(dtype=dtype)

    def initial_history(self, state):
        return state.to(self.state_mean.dtype).unsqueeze(-2).expand(*state.shape[:-1],self.context,5).clone()

    def advance_history(self, history, state):
        return torch.cat((history[...,1:,:],state.to(self.state_mean.dtype).unsqueeze(-2)),-2)

    def core_delta(self, state, history=None):
        history = self.initial_history(state) if history is None else history.to(self.state_mean.dtype)
        embedded = self.backbone((history-self.state_mean)/self.state_std)[...,-1,:]
        return self.core_head(embedded)*self.delta_scale+self.delta_mean

    def contact_features(self, state):
        f = (state.to(self.state_mean.dtype)-self.state_mean)/self.state_std
        return f[...,:2] if self.gate_input_dim == 2 else f

    def raw_gate_logits(self, state):
        f = self.contact_features(state)
        logits=self.contact_head(self.contact_encoder(f))
        return logits+self.contact_linear(f) if self.config.get('linear_gate_enabled',True) else logits

    def gate_logits(self, state):
        logits=self.raw_gate_logits(state)
        if self.modes==1 and self.gate_modes==2:
            return logits.amax(-1,keepdim=True) if self.config.get('gate_pool','logsumexp')=='max' else logits.logsumexp(-1,keepdim=True)
        return logits

    def bounce_features(self, state, kind=0):
        s = state.to(self.state_mean.dtype)
        f = s[...,2:4] if self.bounce_input_dim == 2 else s[...,2:] if self.bounce_input_dim == 3 else s
        return (f-self.bounce_mean[kind])/self.bounce_std[kind]

    def responses(self, state):
        return torch.stack([(self.bounce[k](self.bounce_features(state,k))+
            self.bounce_linear[k](self.bounce_features(state,k)))*self.response_std[k]+self.response_mean[k]
            for k in range(self.modes)],-2)

    def components(self, state, history=None):
        delta1 = self.core_delta(state,history)
        logits = self.gate_logits(state)
        probabilities = logits.sigmoid()
        threshold = float(self.config.get('gate_threshold',.5))
        gates = (logits >= math.log(threshold/(1-threshold))).to(delta1.dtype)
        if self.config.get('gate','hard') == 'soft': gates = probabilities
        if not self.contact_enabled: gates = torch.zeros_like(gates)
        return dict(delta1=delta1,delta2_by_mode=self.responses(state),logits=logits,
                    probabilities=probabilities,gates=gates)

    def details(self, state, history=None):
        s = state.to(self.state_mean.dtype)
        info = self.components(s,history)
        return s+info['delta1']+(info['gates'][...,None]*info['delta2_by_mode']).sum(-2), info

    def forward(self, state, history=None):
        return self.details(state,history)[0]

    def rollout(self, initial, steps):
        state = initial.to(self.state_mean.dtype)
        history = self.initial_history(state)
        trajectory = [state]
        for _ in range(steps):
            state = self(state,history)
            history = self.advance_history(history,state)
            trajectory.append(state)
        return torch.stack(trajectory,-2)


def truth_history(states, pairs, context):
    offsets = torch.arange(1-context,1,device=states.device)
    times = (pairs[:,1,None]+offsets).clamp_min(0)
    return states[pairs[:,0,None],times]


def load_temporal(path,device='cpu'):
    packet = torch.load(path,map_location=device,weights_only=False)
    model = TemporalContactNRD(packet['model_config'],packet['normalization']).to(device)
    model.load_state_dict(packet['model_state_dict'])
    return model.eval(),packet
