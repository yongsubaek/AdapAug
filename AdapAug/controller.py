import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from AdapAug.networks.resnet import ResNet
from theconf import Config as C

class Controller(nn.Module):
    def __init__(self,
                 n_subpolicy=5,
                 n_op=2,
                 emb_size=32,
                 lstm_size=100,
                 operation_types = 15,
                 operation_prob = 11,
                 operation_mag = 10,
                 lstm_num_layers=1,
                 baseline=None,
                 tanh_constant=1.5,
                 temperature=None,
                 img_input=True,
                 group_fnc=None,
                 n_controller=1,
                 ):
        super(Controller, self).__init__()

        self.n_subpolicy = n_subpolicy
        self.n_op = n_op
        self.emb_size = emb_size
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.baseline = baseline
        self.tanh_constant = tanh_constant
        self.temperature = temperature

        self._operation_types = operation_types
        self._operation_prob = operation_prob
        self._operation_mag = operation_mag

        self.img_input = img_input
        self.group_fnc = group_fnc
        self.n_controller = n_controller
        self._create_params()

    def _create_params(self):
        self.lstm = nn.ModuleList([
            nn.LSTM(input_size=self.emb_size,
                    hidden_size=self.lstm_size,
                    num_layers=self.lstm_num_layers)
            for _ in range(self.n_controller)
        ])

        # self.lstm = nn.LSTM(input_size=self.emb_size,
        #                     hidden_size=self.lstm_size,
        #                     num_layers=self.lstm_num_layers)
        
        if self.img_input:
            # input CNN
            # self.conv_input = nn.Sequential(
            #     # Input size: [batch, 3, 32, 32]
            #     # Output size: [1, batch, lstm_size]
            #     nn.Conv2d(3, 16, 3, stride=2, padding=1),            # [batch, 16, 16, 16]
            #     nn.ReLU(),
            #     nn.Conv2d(16, 32, 3, stride=2, padding=1),           # [batch, 32, 8, 8]
            #     nn.BatchNorm2d(32),
            #     nn.ReLU(),
    		# 	nn.Conv2d(32, 64, 3, stride=2, padding=1),           # [batch, 64, 4, 4]
            #     nn.BatchNorm2d(64),
            #     nn.ReLU(),
            #     nn.AvgPool2d(2, stride=2),                            # [batch, 64, 2, 2]
            #     nn.Flatten(),
            #     nn.Linear(64*2*2, self.emb_size)
            # )
            self.conv_input = ResNet(dataset=C.get()['dataset'], depth=18, num_classes=self.emb_size, bottleneck=True)
        else:
            if self.group_fnc is None:
                self.in_emb = nn.Embedding(1, self.emb_size)  # Learn the starting input
            else:
                self.in_emb = nn.Embedding(2, self.emb_size)  # Learn the starting input

        # LSTM output to Categorical logits
        self.o_logit = nn.ModuleList([
            nn.Linear(self.lstm_size, self._operation_types)#, bias=False)
            for _ in range(self.n_controller)
        ])
        self.p_logit = nn.ModuleList([
            nn.Linear(self.lstm_size, self._operation_prob )#, bias=False)
            for _ in range(self.n_controller)
        ])
        self.m_logit = nn.ModuleList([
            nn.Linear(self.lstm_size, self._operation_mag  )#, bias=False)
            for _ in range(self.n_controller)
        ])
        # Embedded input to LSTM: (class:int)->(lstm input vector)
        self.o_emb = nn.ModuleList([
            nn.Embedding(self._operation_types, self.emb_size)
            for _ in range(self.n_controller)
        ])
        self.p_emb = nn.ModuleList([
            nn.Embedding(self._operation_prob , self.emb_size)
            for _ in range(self.n_controller)
        ])
        self.m_emb = nn.ModuleList([
            nn.Embedding(self._operation_mag  , self.emb_size)
            for _ in range(self.n_controller)
        ])
        
        # # LSTM output to Categorical logits
        # self.o_logit = nn.Linear(self.lstm_size, self._operation_types)#, bias=False)
        # self.p_logit = nn.Linear(self.lstm_size, self._operation_prob )#, bias=False)
        # self.m_logit = nn.Linear(self.lstm_size, self._operation_mag  )#, bias=False)
        # # Embedded input to LSTM: (class:int)->(lstm input vector)
        # self.o_emb = nn.Embedding(self._operation_types, self.emb_size)
        # self.p_emb = nn.Embedding(self._operation_prob , self.emb_size)
        # self.m_emb = nn.Embedding(self._operation_mag  , self.emb_size)

        self.softmax = nn.Softmax(dim=1)

        self._reset_params()

    def _reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)
        for i_ctr in range(self.n_controller):
            nn.init.uniform_(self.lstm[i_ctr].weight_hh_l0, -0.1, 0.1)
            nn.init.uniform_(self.lstm[i_ctr].weight_ih_l0, -0.1, 0.1)

    def rnn_params(self):
        # return nn.ParameterList([ param for param in self.parameters() if param not in self.conv_input.parameters()])
        ctl_params = nn.ParameterList(
                    list(self.lstm.parameters()) +
                    list(self.o_logit.parameters()) +
                    list(self.p_logit.parameters()) +
                    list(self.m_logit.parameters()) +
                    list(self.o_emb.parameters()) +
                    list(self.p_emb.parameters()) +
                    list(self.m_emb.parameters())
                    )
        if not self.img_input:
            ctl_params.extend(self.in_emb.parameters())
        return ctl_params

    def softmax_tanh(self, logit):
        if self.temperature is not None:
            logit /= self.temperature
        if self.tanh_constant is not None:
            logit = self.tanh_constant * torch.tanh(logit)
        return logit

    def forward(self, image=None, policy=None, label=None):
        """
        return: log_probs, entropys, subpolicies
        log_probs: batch of log_prob, (tensor)[batch or 1]
        entropys: batch of entropy, (tensor)[batch or 1]
        subpolicies: batch of sampled policies, (np.array)[batch, n_subpolicy, n_op, 3]
        """      
        batch_size = image.shape[0]
        
        if self.img_input:
            inputs = self.conv_input(image)                 # [batch, lstm_size]
            index_list = [list(range(inputs.shape[0]))]
        else:
            if self.group_fnc is None:
                inputs = self.in_emb.weight                  # [1, lstm_size]
                index_list = [list(range(inputs.shape[0]))]
            else:
                groups = self.group_fnc[label]
                inputs = self.in_emb.weight[groups]          # [batch, lstm_size]
                index_list = [list(range(inputs.shape[0]))]

                if self.n_controller > 1:
                    index_list = []
                    index_list.append([i for i, grp in enumerate(groups) if grp == 0])
                    index_list.append([i for i, grp in enumerate(groups) if grp == 1])                    
        inputs_list = [inputs[index].unsqueeze(0) for index in index_list] # [n_controller, 1, batch_chunk(or 1), lstm_size]
        controller_subpolicy = []
        log_probs_list = []
        entropys_list = []
        for i_ctr in range(self.n_controller):
            if len(index_list[i_ctr]) == 0:
                controller_subpolicy.append(torch.tensor([]).cuda())
                log_probs_list.append(torch.tensor([]).cuda())
                entropys_list.append(torch.tensor([]).cuda())
                continue
            self.hidden = None  # setting state to None will initialize LSTM state with 0s
            subpolicies = []
            log_probs = []
            entropys = []
            for i_subpol in range(self.n_subpolicy):
                subpolicy = []
                for i_op in range(self.n_op):
                    # sample operation type, o
                    output, self.hidden = self.lstm[i_ctr](inputs_list[i_ctr], self.hidden)        # [1, batch, lstm_size]
                    output = output.squeeze(0)                      # [batch, lstm_size]
                    logit = self.o_logit[i_ctr](output)                    # [batch, _operation_types]
                    logit = self.softmax_tanh(logit)
                    o_id_dist = Categorical(logits=logit)
                    if policy is not None:
                        i_policy = policy[index_list[i_ctr]]
                        o_id = i_policy[:,i_subpol,i_op,0]
                    else:
                        o_id = o_id_dist.sample()                   # [batch]
                    log_prob = o_id_dist.log_prob(o_id)             # [batch]
                    entropy = o_id_dist.entropy()                   # [batch]
                    log_probs.append(log_prob)
                    entropys.append(entropy)
                    inputs_list[i_ctr] = self.o_emb[i_ctr](o_id)                       # [batch, lstm_size]
                    inputs_list[i_ctr] = inputs_list[i_ctr].unsqueeze(0)                    # [1, batch, lstm_size]
                    if self._operation_prob > 0:
                        # sample operation probability, p
                        output, self.hidden = self.lstm[i_ctr](inputs_list[i_ctr], self.hidden)
                        output = output.squeeze(0)
                        logit = self.p_logit[i_ctr](output)
                        logit = self.softmax_tanh(logit)
                        p_id_dist = Categorical(logits=logit)
                        if policy is not None:
                            i_policy = policy[index_list[i_ctr]]
                            p_id = i_policy[:,i_subpol,i_op,1]
                        else:
                            p_id = p_id_dist.sample()
                        log_prob = p_id_dist.log_prob(p_id)
                        entropy = p_id_dist.entropy()
                        log_probs.append(log_prob)
                        entropys.append(entropy)
                        inputs_list[i_ctr] = self.p_emb[i_ctr](p_id)
                        inputs_list[i_ctr] = inputs_list[i_ctr].unsqueeze(0)
                    else:
                        p_id = 10 * torch.ones_like(o_id).cuda()
                    # sample operation magnitude, m
                    output, self.hidden = self.lstm[i_ctr](inputs_list[i_ctr], self.hidden)
                    output = output.squeeze(0)
                    logit = self.m_logit[i_ctr](output)
                    logit = self.softmax_tanh(logit)
                    m_id_dist = Categorical(logits=logit)
                    if policy is not None:
                        i_policy = policy[index_list[i_ctr]]
                        m_id = i_policy[:,i_subpol,i_op,2]
                    else:
                        m_id = m_id_dist.sample()
                    log_prob = m_id_dist.log_prob(m_id)
                    entropy = m_id_dist.entropy()
                    log_probs.append(log_prob)
                    entropys.append(entropy)
                    inputs_list[i_ctr] = self.m_emb[i_ctr](m_id)
                    inputs_list[i_ctr] = inputs_list[i_ctr].unsqueeze(0)
                    subpolicy.append([o_id.detach().cpu().numpy(), p_id.detach().cpu().numpy(), m_id.detach().cpu().numpy()])
                subpolicies.append(subpolicy)
            subpolicies = np.array(subpolicies)                   # (np.array) [n_subpolicy, n_op, 3, batch_chunk]
            subpolicies = torch.from_numpy(np.moveaxis(subpolicies,-1,0)).cuda()  # [batch_chunk, n_subpolicy, n_op, 3]

            controller_subpolicy.append(subpolicies)
            log_probs_list.append(sum(log_probs)) # (tensor) [batch_chunk]
            entropys_list.append(sum(entropys)) # (tensor) [batch_chunk]

        sampled_policies = controller_subpolicy                   # (list) [n_controller, batch_chunk (different), n_subpolicy, n_op, 3]
        zeros_polices = torch.zeros([batch_size, self.n_subpolicy, self.n_op, 3], dtype=torch.int64).cuda() # [batch, n_subpolicy, n_op, 3]
        zeros_log_probs = torch.zeros([batch_size], dtype=torch.float32).cuda()
        zeros_entropys = torch.zeros([batch_size], dtype=torch.float32).cuda()
        for i_ctr in range(self.n_controller):
            if len(sampled_policies[i_ctr]) > 0:
                zeros_polices[index_list[i_ctr]] = sampled_policies[i_ctr]
                zeros_log_probs[index_list[i_ctr]] = log_probs_list[i_ctr]
                zeros_entropys[index_list[i_ctr]] = entropys_list[i_ctr]

        sampled_policies = zeros_polices
        log_probs = zeros_log_probs                            # (tensor) [batch]
        entropys = zeros_entropys                              # (tensor) [batch]
        return log_probs, entropys, sampled_policies

class RandAug(object):
    """
    """
    def __init__(self,
                 n_subpolicy=2,
                 n_op=2,
                 operation_types = 15,
                 operation_prob = 11,
                 operation_mag = 10):
        self.n_subpolicy = n_subpolicy
        self.n_op = n_op
        self._operation_types = operation_types
        self._operation_prob = operation_prob
        self._operation_mag = operation_mag
        self._search_space_size = [self._operation_types, self._operation_prob, self._operation_mag]

    def __call__(self, input):
        """
        input: (tensor) [batch, W, H, 3]
        return sampled_policies: (np.array) [batch, n_subpolicy, n_op, 3]
        """
        # *_id (np.array) [batch]
        batch_size = input.size(0)
        subpolicies = []
        for i_subpol in range(self.n_subpolicy):
            subpolicy = []
            for i_op in range(self.n_op):
                oper = []
                for oper_len in self._search_space_size:
                    ids = np.random.randint(0, oper_len, batch_size)
                    oper.append(ids)
                subpolicy.append(oper)
            subpolicies.append(subpolicy)
        sampled_policies = np.array(subpolicies)                    # (np.array) [n_subpolicy, n_op, 3, batch]
        self.sampled_policies = np.moveaxis(sampled_policies,-1,0)  # (np.array) [batch, n_subpolicy, n_op, 3]
        return None, None, self.sampled_policies

    def eval(self):
        pass

    def train(self):
        pass
