import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from . import model_utils
from .model_utils import RefNRIMLP, encode_onehot
import math


class DNRI(nn.Module):
    def __init__(self, params):
        super(DNRI, self).__init__()
        # Model Params
        self.model_type = params['model_type']
        self.num_vars = params['num_vars']
        self.encoder = DNRI_Encoder(params)
        decoder_type = params.get('decoder_type', None)
        if decoder_type == 'ref_mlp':
            self.decoder = DNRI_MLP_Decoder(params)
        else:
            self.decoder = DNRI_Decoder(params)
        self.num_edge_types = params.get('num_edge_types')

        # Training params
        self.layer_num_edge_types = params.get('layer_num_edge_types')
        self.gumbel_temp = params.get('gumbel_temp')
        self.train_hard_sample = params.get('train_hard_sample')
        self.teacher_forcing_steps = params.get('teacher_forcing_steps', -1)
        
        self.normalize_kl = params.get('normalize_kl', False)
        self.normalize_kl_per_var = params.get('normalize_kl_per_var', False)
        self.normalize_nll = params.get('normalize_nll', False)
        self.normalize_nll_per_var = params.get('normalize_nll_per_var', False)
        self.kl_coef = params.get('kl_coef', 1.)
        self.nll_loss_type = params.get('nll_loss_type', 'crossent')
        self.prior_variance = params.get('prior_variance')
        self.timesteps = params.get('timesteps', 0)
        self.burn_in_steps = params.get('train_burn_in_steps')
        self.teacher_forcing_prior = params.get('teacher_forcing_prior', False)
        self.val_teacher_forcing_steps = params.get('val_teacher_forcing_steps', -1)
        self.add_uniform_prior = params.get('add_uniform_prior')
        if self.add_uniform_prior:
            if params.get('no_edge_prior') is not None:
                prior = np.zeros(self.num_edge_types)
                prior.fill((1 - params['no_edge_prior'])/(self.num_edge_types - 1))
                prior[0] = params['no_edge_prior']
                log_prior = torch.FloatTensor(np.log(prior))
                log_prior = torch.unsqueeze(log_prior, 0)
                log_prior = torch.unsqueeze(log_prior, 0)
                if params['gpu']:
                    log_prior = log_prior.cuda(non_blocking=True)
                self.log_prior = log_prior
                print("USING NO EDGE PRIOR: ",self.log_prior)
            else:
                print("USING UNIFORM PRIOR")
                prior = np.zeros(self.num_edge_types)
                prior.fill(1.0/self.num_edge_types)
                log_prior = torch.FloatTensor(np.log(prior))
                log_prior = torch.unsqueeze(log_prior, 0)
                log_prior = torch.unsqueeze(log_prior, 0)
                if params['gpu']:
                    log_prior = log_prior.cuda(non_blocking=True)
                self.log_prior = log_prior

    def single_step_forward(self, inputs, decoder_hidden, edge_logits, hard_sample):
        old_shape = edge_logits.shape # shape: [batch_size, num_edges, num_edge_types] (=[4, 930, 4])
        if self.model_type == 'dnri':
            # Edge type vectors z^{t}_{(i, j)} are sampled for all edges (i, j)
            # according to eq. (13) or (15) depending on args.mode ('eval' or 'train').
            edges = model_utils.gumbel_softmax(
                edge_logits.reshape(-1, self.num_edge_types), 
                tau=self.gumbel_temp, hard=hard_sample
            ).view(old_shape)
        else:
            # Edge type vectors z^{t, a}_{(i, j)} are sampled for all edges (i, j) and all layer-graphs 1<=a<=n.
            # See eq. (4) to (6) in the fNRI paper.
            layer_num_edge_types = self.layer_num_edge_types
            edges = []
            for i in range(len(layer_num_edge_types)):
                start_segment = sum(layer_num_edge_types[:i])
                end_segment = start_segment + layer_num_edge_types[i]
                # See eq. (4) in fNRI paper
                segment_logits = edge_logits[:, :, start_segment:end_segment]
                # See eq. (5) in fNRI paper
                segment_edges = model_utils.gumbel_softmax(
                    segment_logits.reshape(-1, layer_num_edge_types[i]), 
                    tau=self.gumbel_temp, hard=hard_sample
                )
                edges.append(segment_edges)
            edges = torch.cat(edges, dim=-1).view(old_shape)
            
            # Check if an edge type is chosen in every layer-graph when using hard samples.
            if hard_sample:
                assert (torch.eq(torch.sum(edges, dim=-1), float(len(self.layer_num_edge_types)))).all()
        
        # predictions = \mu^{t+1}_{j} for all nodes j, see eq. (16) in NRI paper
        # predictions.shape = [batch_size, num_vars, input_size] (=[4, 31, 6])
        # decoder_hidden = \tilde{h}^{t+1}_j for all nodes j, see eq. (16) in NRI paper
        # decoder_hidden.shape = [batch_size, num_vars, rnn_hidden] (=[4, 31, 256]) where rnn_hidden = params['decoder_hidden']
        predictions, decoder_hidden = self.decoder(inputs, decoder_hidden, edges)
        return predictions, decoder_hidden, edges

    def calculate_loss(self, inputs, is_train=False, teacher_forcing=True, return_edges=False, return_logits=False, use_prior_logits=False):
        # inputs.shape = [batch_size, num_timesteps, num_nodes, input_size] (=[4, 50, 31, 6])
        decoder_hidden = self.decoder.get_initial_hidden(inputs) # shape: [batch_size, num_nodes, rnn_hidden] (=[4, 31, 256]) where rnn_hidden = params['decoder_hidden']
        num_time_steps = inputs.size(1)
        all_edges = []
        all_predictions = []
        all_priors = []
        hard_sample = (not is_train) or self.train_hard_sample
        # Note that the encoder doesn't take the last time step as input during training 
        # because the decoder only needs z^{1:T-1} in order to fully reconstruct the input x^{1:T}.
        # See eq. (6) in the dNRI paper.
        # prior_logits.shape = [batch_size, num_timesteps, num_edges, num_edge_types] (=[4, 49, 930, 4])
        # posterior_logits.shape = [batch_size, num_timesteps, num_edges, num_edge_types] (=[4, 49, 930, 4])
        prior_logits, posterior_logits, _ = self.encoder(inputs[:, :-1])
        if not is_train:
            teacher_forcing_steps = self.val_teacher_forcing_steps
        else:
            teacher_forcing_steps = self.teacher_forcing_steps
        for step in range(num_time_steps-1):
            # step = t
            # teacher_forcing refers to the process described in the last paragraph of section 3.5 in the NRI paper.
            if (teacher_forcing and (teacher_forcing_steps == -1 or step < teacher_forcing_steps)) or step == 0:
                # current_inputs = x^{t}
                current_inputs = inputs[:, step] # shape: [batch_size, num_nodes, input_size] (=[4, 31, 6])
            else:
                current_inputs = predictions
            if not use_prior_logits:
                # current_p_logits = f_{enc}([h^{t}_{(i,j),enc}, h^{t}_{(i,j),prior}]) for all edges (i, j)
                current_p_logits = posterior_logits[:, step] # shape: [batch_size, num_edges, num_edge_types] (=[4, 930, 4])
            else:
                current_p_logits = prior_logits[:, step]
            # decoder_hidden = \tilde{h}^{t+1}_{j} for all nodes j, see equation (15) in NRI paper
            predictions, decoder_hidden, edges = self.single_step_forward(current_inputs, decoder_hidden, current_p_logits, hard_sample)
            # all_predictions = [\mu^{2}, ... , \mu^{t+1}], see eq. (16) in NRI paper
            all_predictions.append(predictions)
            all_edges.append(edges)
        # all_predictions = [\mu^{2}, ... , \mu^{T}], see eq. (16) in NRI paper
        all_predictions = torch.stack(all_predictions, dim=1) # shape: [batch_size, num_timesteps-1, num_nodes, input_size] (=[4, 49, 31, 6])
        # all_predictions = [x^{2}, ... , x^{T}]
        target = inputs[:, 1:, :, :]
        # loss_nll = reconstruction error described by eq. (18) in NRI paper
        loss_nll = self.nll(all_predictions, target)
        if self.model_type == 'dnri':
            # See eq. (15) in dNRI paper
            prob = F.softmax(posterior_logits, dim=-1)
            # See eq. (16) in dNRI paper
            loss_kl = self.kl_categorical_learned(prob, prior_logits)
        else:
            # Compute KL divergence segment-wise as in eq. (7) in the fNRI paper.
            posterior_logits = torch.split(posterior_logits, self.layer_num_edge_types, dim = -1)
            prior_logits = torch.split(prior_logits, self.layer_num_edge_types, dim = -1)
            loss_kl = 0
            prob = []
            for i in range(len(self.layer_num_edge_types)):
                prob.append(F.softmax(posterior_logits[i], dim = -1))
                loss_kl += self.kl_categorical_learned(prob[i], prior_logits[i])
        
        # CHECK: The case below doesn't occur for the default experiments.
        if self.add_uniform_prior:
            if self.model_type == 'dfnri':
                raise NotImplementedError('Not implemented for dfNRI.')
            loss_kl = 0.5*loss_kl + 0.5*self.kl_categorical_avg(prob)
        
        loss = loss_nll + self.kl_coef*loss_kl
        loss = loss.mean()

        if return_edges:
            return loss, loss_nll, loss_kl, edges
        elif return_logits:
            return loss, loss_nll, loss_kl, posterior_logits, all_predictions
        else:
            return loss, loss_nll, loss_kl

    def predict_future(self, inputs, prediction_steps, return_edges=False, return_everything=False):
        burn_in_timesteps = inputs.size(1)# number of timesteps of the training data
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        all_predictions = []
        all_edges = []
        prior_logits, _, prior_hidden = self.encoder(inputs[:, :-1])# compute the prior p_{phi} for all inputs using a forward LSTM.
        for step in range(burn_in_timesteps-1):
            current_inputs = inputs[:, step] # prvide ground-truth states to the decoder as input during training, unlike static NRI
            current_edge_logits = prior_logits[:, step]
            predictions, decoder_hidden, edges = self.single_step_forward(current_inputs, decoder_hidden, current_edge_logits, True)
            if return_everything:
                all_edges.append(edges)
                all_predictions.append(predictions)
        predictions = inputs[:, burn_in_timesteps-1]
        for step in range(prediction_steps):
            # predicting future states fo the system -> can't use encoder to predict edges
            current_edge_logits, prior_hidden = self.encoder.single_step_forward(predictions, prior_hidden)
            # given previous predictions, compute prior distribution p_{phi}, sample from that prior (current_edge_logits) to obtain current relations predictions
            predictions, decoder_hidden, edges = self.single_step_forward(predictions, decoder_hidden, current_edge_logits, True)
            all_predictions.append(predictions)
            all_edges.append(edges)
        
        predictions = torch.stack(all_predictions, dim=1)
        if return_edges:
            edges = torch.stack(all_edges, dim=1)
            return predictions, edges
        else:
            return predictions

    def copy_states(self, state):
        if isinstance(state, tuple) or isinstance(state, list):
            current_state = (state[0].clone(), state[1].clone())
        else:
            current_state = state.clone()
        return current_state

    def merge_hidden(self, hidden):
        if isinstance(hidden[0], tuple) or isinstance(hidden[0], list):
            result0 = torch.cat([x[0] for x in hidden], dim=0)
            result1 = torch.cat([x[1] for x in hidden], dim=0)
            return (result0, result1)
        else:
            return torch.cat(hidden, dim=0)

    def predict_future_fixedwindow(self, inputs, burn_in_steps, prediction_steps, batch_size, return_edges=False):
        print("INPUT SHAPE: ",inputs.shape)
        prior_logits, _, prior_hidden = self.encoder(inputs[:, :-1])
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        for step in range(burn_in_steps-1):
            current_inputs = inputs[:, step]
            current_edge_logits = prior_logits[:, step]
            predictions, decoder_hidden, _ = self.single_step_forward(current_inputs, decoder_hidden, current_edge_logits, True)
        all_timestep_preds = []
        all_timestep_edges = []
        for window_ind in range(burn_in_steps - 1, inputs.size(1)-1, batch_size):
            current_batch_preds = []
            current_batch_edges = []
            prior_states = []
            decoder_states = []
            for step in range(batch_size):
                if window_ind + step >= inputs.size(1):
                    break
                predictions = inputs[:, window_ind + step] 
                current_edge_logits, prior_hidden = self.encoder.single_step_forward(predictions, prior_hidden)
                predictions, decoder_hidden, _ = self.single_step_forward(predictions, decoder_hidden, current_edge_logits, True)
                current_batch_preds.append(predictions)
                tmp_prior = self.encoder.copy_states(prior_hidden)
                tmp_decoder = self.copy_states(decoder_hidden)
                prior_states.append(tmp_prior)
                decoder_states.append(tmp_decoder)
                if return_edges:
                    current_batch_edges.append(current_edge_logits.cpu())
            batch_prior_hidden = self.encoder.merge_hidden(prior_states)
            batch_decoder_hidden = self.merge_hidden(decoder_states)
            current_batch_preds = torch.cat(current_batch_preds, 0)
            current_timestep_preds = [current_batch_preds]
            if return_edges:
                current_batch_edges = torch.cat(current_batch_edges, 0)
                current_timestep_edges = [current_batch_edges]
            for step in range(prediction_steps - 1):
                current_batch_edge_logits, batch_prior_hidden = self.encoder.single_step_forward(current_batch_preds, batch_prior_hidden)
                current_batch_preds, batch_decoder_hidden, _ = self.single_step_forward(current_batch_preds, batch_decoder_hidden, current_batch_edge_logits, True)
                current_timestep_preds.append(current_batch_preds)
                if return_edges:
                    current_timestep_edges.append(current_batch_edge_logits.cpu())
            all_timestep_preds.append(torch.stack(current_timestep_preds, dim=1))
            if return_edges:
                all_timestep_edges.append(torch.stack(current_timestep_edges, dim=1))
        result =  torch.cat(all_timestep_preds, dim=0)
        if return_edges:
            edge_result = torch.cat(all_timestep_edges, dim=0)
            return result.unsqueeze(0), edge_result.unsqueeze(0)
        else:
            return result.unsqueeze(0)

    def nll(self, preds, target):
        if self.nll_loss_type == 'crossent':
            return self.nll_crossent(preds, target)
        elif self.nll_loss_type == 'gaussian':
            return self.nll_gaussian(preds, target)
        elif self.nll_loss_type == 'poisson':
            return self.nll_poisson(preds, target)

    def nll_gaussian(self, preds, target, add_const=False):
        neg_log_p = ((preds - target) ** 2 / (2 * self.prior_variance))
        const = 0.5 * np.log(2 * np.pi * self.prior_variance)
        #neg_log_p += const
        if self.normalize_nll_per_var:
            return neg_log_p.sum() / (target.size(0) * target.size(2))
        elif self.normalize_nll:
            return (neg_log_p.sum(-1) + const).view(preds.size(0), -1).mean(dim=1)
        else:
            return neg_log_p.view(target.size(0), -1).sum() / (target.size(1))

    def nll_crossent(self, preds, target):
        if self.normalize_nll:
            return nn.BCEWithLogitsLoss(reduction='none')(preds, target).view(preds.size(0), -1).mean(dim=1)
        else:
            return nn.BCEWithLogitsLoss(reduction='none')(preds, target).view(preds.size(0), -1).sum(dim=1)

    def nll_poisson(self, preds, target):
        if self.normalize_nll:
            return nn.PoissonNLLLoss(reduction='none')(preds, target).view(preds.size(0), -1).mean(dim=1)
        else:
            return nn.PoissonNLLLoss(reduction='none')(preds, target).view(preds.size(0), -1).sum(dim=1)

    def kl_categorical_learned(self, preds, prior_logits):
        log_prior = nn.LogSoftmax(dim=-1)(prior_logits)
        kl_div = preds*(torch.log(preds + 1e-16) - log_prior)
        if self.normalize_kl:     
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            return kl_div.sum() / (self.num_vars * preds.size(0))
        else:
            return kl_div.view(preds.size(0), -1).sum(dim=1)

    def kl_categorical_avg(self, preds, eps=1e-16):
        avg_preds = preds.mean(dim=2)
        kl_div = avg_preds*(torch.log(avg_preds+eps) - self.log_prior)
        if self.normalize_kl:     
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            return kl_div.sum() / (self.num_vars * preds.size(0))
        else:
            return kl_div.view(preds.size(0), -1).sum(dim=1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DNRI_Encoder(nn.Module):
    '''
    DNRI_Encoder implements the prior and encoder described by equations (8)-(15) in the dNRI paper.
    '''
    def __init__(self, params):
        super(DNRI_Encoder, self).__init__()
        num_vars = params['num_vars']
        # CHANGE NECESSARY FOR DFNRI
        # CHECK: self.num_edges might need to be handled differently for dfNRI
        self.num_edges = params['num_edge_types']
        self.sepaate_prior_encoder = params.get('separate_prior_encoder', False)
        no_bn = False
        dropout = params['encoder_dropout']
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()), requires_grad=False)
        self.save_eval_memory = params.get('encoder_save_eval_memory', False)


        hidden_size = params['encoder_hidden'] # (=256)
        rnn_hidden_size = params['encoder_rnn_hidden'] # (=64)
        rnn_type = params['encoder_rnn_type']
        inp_size = params['input_size'] # (=6)
        # self.mlp1 = f_{emb}, see equation (8) in dNRI paper
        self.mlp1 = RefNRIMLP(inp_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        # self.mlp2 = f^{1}_{e}, see equation (9) in dNRI paper
        self.mlp2 = RefNRIMLP(hidden_size * 2, hidden_size, hidden_size, dropout, no_bn=no_bn)
        # self.mlp3 = f^{1}_{v}, see equation (10) in dNRI paper
        self.mlp3 = RefNRIMLP(hidden_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        # self.mlp4 = f^{2}_{e}, see equation (11) in dNRI paper
        # CHECK: Shouldn't the first argument (input size) be: hidden_size * 2?
        self.mlp4 = RefNRIMLP(hidden_size * 3, hidden_size, hidden_size, dropout, no_bn=no_bn)

        if rnn_hidden_size is None:
            rnn_hidden_size = hidden_size
        if rnn_type == 'lstm':
            # self.forward_rnn = LSTM_{prior}, see equation (12) in dNRI paper
            # hidden_size – The number of expected features in the input h^{t}_{(i,j),emb}
            # rnn_hidden_size – The number of features in the hidden state h^{t-1}_{(i,j),prior}
            self.forward_rnn = nn.LSTM(hidden_size, rnn_hidden_size, batch_first=True)
            # self.reverse_rnn = LSTM_{enc}, see equation (14) in dNRI paper
            # hidden_size – The number of expected features in the input h^{t}_{(i,j),emb}
            # rnn_hidden_size – The number of features in the hidden state h^{t+1}_{(i,j),enc}
            self.reverse_rnn = nn.LSTM(hidden_size, rnn_hidden_size, batch_first=True)
        elif rnn_type == 'gru':
            self.forward_rnn = nn.GRU(hidden_size, rnn_hidden_size, batch_first=True)
            self.reverse_rnn = nn.GRU(hidden_size, rnn_hidden_size, batch_first=True)
        # The concatenated vector [h^{t}_{(i,j),enc}, h^{t}_{(i,j),prior}] has size out_hidden_size 
        # and is the input to f_{enc}. See equation (15) in the dNRI paper. 
        out_hidden_size = 2*rnn_hidden_size
        num_layers = params['encoder_mlp_num_layers']
        if num_layers == 1:
            self.encoder_fc_out = nn.Linear(out_hidden_size, self.num_edges)
        else:
            tmp_hidden_size = params['encoder_mlp_hidden']
            # CHECK: According to the dNRI paper, ReLU activations are used, not ELU, see section 4.
            layers = [nn.Linear(out_hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, self.num_edges))
            # self.encoder_fc_out = f_{enc}, see equation (15) in dNRI paper
            self.encoder_fc_out = nn.Sequential(*layers)

        num_layers = params['prior_num_layers']
        if num_layers == 1:
            self.prior_fc_out = nn.Linear(rnn_hidden_size, self.num_edges)
        else:
            tmp_hidden_size = params['prior_hidden_size']
            # CHECK: According to the dNRI paper, ReLU activations are used, not ELU, see section 4.
            layers = [nn.Linear(rnn_hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, self.num_edges))
            # self.prior_fc_out = f_{prior}, see equation (13) in dNRI paper
            self.prior_fc_out = nn.Sequential(*layers)


        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, node_embeddings):
        # Input size: [batch, num_vars, num_timesteps, embed_size]
        if len(node_embeddings.shape) == 4:
            send_embed = node_embeddings[:, self.send_edges, :, :]
            recv_embed = node_embeddings[:, self.recv_edges, :, :]
        else:
            send_embed = node_embeddings[:, self.send_edges, :]
            recv_embed = node_embeddings[:, self.recv_edges, :]
        return torch.cat([send_embed, recv_embed], dim=-1)

    def edge2node(self, edge_embeddings):
        if len(edge_embeddings.shape) == 4:
            old_shape = edge_embeddings.shape
            tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)
            incoming = torch.matmul(self.edge2node_mat, tmp_embeddings).view(old_shape[0], -1, old_shape[2], old_shape[3])
        else:
            incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming/(self.num_vars-1) #TODO: do we want this average?

    def copy_states(self, prior_state):
        if isinstance(prior_state, tuple) or isinstance(prior_state, list):
            current_prior_state = (prior_state[0].clone(), prior_state[1].clone())
        else:
            current_prior_state = prior_state.clone()
        return current_prior_state

    def merge_hidden(self, hidden):
        if isinstance(hidden[0], tuple) or isinstance(hidden[0], list):
            result0 = torch.cat([x[0] for x in hidden], dim=0)
            result1 = torch.cat([x[1] for x in hidden], dim=0)
            result = (result0, result1)
        else:
            result = torch.cat(hidden, dim=0)
        return result

    def forward(self, inputs):
        if self.training or not self.save_eval_memory:
            # inputs.shape = [batch_size, num_timesteps, num_vars, input_size] (=[4, 49, 31, 6])
            # CHECK: num_timesteps = 49, not 50 as described in section 4.2 of dNRI paper
            num_timesteps = inputs.size(1)
            # x.shape = [batch_size, num_vars, num_timesteps, input_size]
            x = inputs.transpose(2, 1).contiguous()
            # For t in 1:T:
            #   Compute eq. (8) in dNRI paper: h^{t}_{i,1} = f_{emb}(x^{t}_i) ∀ 1<=i<=N, 
            #   i.e. for all nodes in the graph.
            x = self.mlp1(x)  # shape: [batch_size, num_vars, num_timesteps, hidden_size] (=[4, 31, 49, 256]) where hidden_size = params['encoder_hidden']
            # self.node2edge prepares the inputs [h^{t}_{i,1}, h^{t}_{j,1}] for f^{1}_{e} for all edges. 
            # See eq. (9) in dNRI paper.
            x = self.node2edge(x) # shape: [batch_size, num_edges, num_timesteps, 2*hidden_size] (=[4, 930, 49, 512])
            # For t in 1:T:
            #   Compute eq. (9) in dNRI paper: h^{t}_{(i,j),1} = f^{1}_{e}([h^{t}_{i,1}, h^{t}_{j,1}])
            #   ∀ 1<=i,j<=N ∧ i≠j, i.e. for all edges in the graph.
            x = self.mlp2(x) # shape: [batch_size, num_edges, num_timesteps, hidden_size] (=[4, 930, 49, 256])
            # CHECK: Why is the line below needed?
            x_skip = x
            # self.edge2node prepares the inputs \sum_{1<=i<=N ∧ i≠j} h^{t}_{(i,j),1} for f^{1}_{v} for all nodes j. 
            # See eq. (10) in dNRI paper.
            x = self.edge2node(x) # shape: [batch_size, num_vars, num_timesteps, hidden_size] (=[4, 31, 49, 256])
            # For t in 1:T:
            #   Compute eq. (10) in dNRI paper: h^{t}_{j,2} = f^{1}_{v}(\sum_{1<=i<=N ∧ i≠j} h^{t}_{(i,j),1})
            #   ∀ 1<=j<=N, i.e. for all nodes in the graph.
            x = self.mlp3(x) # shape: [batch_size, num_vars, num_timesteps, hidden_size] (=[4, 31, 49, 256])
            # self.node2edge prepares the inputs [h^{t}_{i,2}, h^{t}_{j,2}] for f^{2}_{e} for all edges. 
            # See eq. (11) in dNRI paper.
            x = self.node2edge(x) # shape: [batch_size, num_edges, num_timesteps, 2*hidden_size] (=[4, 930, 49, 512])
            x = torch.cat((x, x_skip), dim=-1)  # Skip connection, shape: [batch_size, num_edges, num_timesteps, 3*hidden_size]
            # CHECK: The input to f^{2}_{e} is different than specified in equation (11) in the dNRI paper.
            # For t in 1:T: 
            #   Compute h^{t}_{(i,j),emb} = f^{2}_{e}([h^{t}_{i,2}, h^{t}_{j,2}, h^{t}_{(i,j),1}])
            #   ∀ 1<=i,j<=N ∧ i≠j, i.e. for all edges in the graph.
            x = self.mlp4(x) # shape: [batch_size, num_edges, num_timesteps, hidden_size] (=[4, 930, 49, 256])
        
            
            # At this point, x should be [batch_size, num_edges, num_timesteps, hidden_size]
            # RNN aggregation
            old_shape = x.shape
            x = x.contiguous().view(-1, old_shape[2], old_shape[3]) # shape: [batch_size * num_edges, num_timesteps, hidden_size]
            # For t in 1:T: 
            #   Compute eq. (12) in dNRI paper: h^{t}_{(i,j),prior} 
            #   ∀ 1<=i,j<=N ∧ i≠j, i.e. for all edges in the graph.
            # forward_x.shape = [batch_size * num_edges, num_timesteps, rnn_hidden_size] (=[3720, 49, 64]) where rnn_hidden_size = params['encoder_rnn_hidden']
            # prior_state[0].shape = [1, 3720, 64]
            # prior_state[1].shape = [1, 3720, 64]
            forward_x, prior_state = self.forward_rnn(x)
            timesteps = old_shape[2]
            reverse_x = x.flip(1) # shape: [batch_size * num_edges, num_timesteps, hidden_size]
            # For t in T:1: 
            #   Compute eq. (14) in dNRI paper: h^{t}_{(i,j),enc} 
            #   ∀ 1<=i,j<=N ∧ i≠j, i.e. for all edges in the graph.
            # reverse_x.shape = [batch_size * num_edges, num_timesteps, rnn_hidden_size] (=[3720, 49, 64]) where rnn_hidden_size = params['encoder_rnn_hidden']
            reverse_x, _ = self.reverse_rnn(reverse_x)
            reverse_x = reverse_x.flip(1)
            
            # forward_x.shape = [batch_size * num_edges, num_timesteps, rnn_hidden_size]
            # For t in 1:T: 
            #   Compute f_{prior}(h^{t}_{(i,j),prior}) as in eq. (13) in dNRI paper
            #   ∀ 1<=i,j<=N ∧ i≠j, i.e. for all edges in the graph.
            # prior_result.shape = [batch_size, num_timesteps, num_edges, num_edge_types] (=[4, 49, 930, 4])
            prior_result = self.prior_fc_out(forward_x).view(old_shape[0], old_shape[1], timesteps, self.num_edges).transpose(1,2).contiguous()
            # See eq. (14)
            combined_x = torch.cat([forward_x, reverse_x], dim=-1)
            # For t in 1:T: 
            #   Compute f_{enc}([h^{t}_{(i,j),enc}, h^{t}_{(i,j),prior}]) as in eq. (15) in dNRI paper
            #   ∀ 1<=i,j<=N ∧ i≠j, i.e. for all edges in the graph.
            # encoder_result.shape = [batch_size, num_timesteps, num_edges, num_edge_types] (=[4, 49, 930, 4])
            encoder_result = self.encoder_fc_out(combined_x).view(old_shape[0], old_shape[1], timesteps, self.num_edges).transpose(1,2).contiguous()
            return prior_result, encoder_result, prior_state
        else:
            # Inputs is shape [batch_size, num_timesteps, num_vars, input_size]
            num_timesteps = inputs.size(1)
            all_x = []
            all_forward_x = []
            all_prior_result = []
            prior_state = None
            for timestep in range(num_timesteps):
                x = inputs[:, timestep]
                #x = inputs.transpose(2, 1).contiguous()
                x = self.mlp1(x)  # 2-layer ELU net per node
                x = self.node2edge(x)
                x = self.mlp2(x)
                x_skip = x
                x = self.edge2node(x)
                x = self.mlp3(x)
                x = self.node2edge(x)
                x = torch.cat((x, x_skip), dim=-1)  # Skip connection
                x = self.mlp4(x)
            
                
                # At this point, x should be [batch_size, num_edges, num_timesteps, hidden_size]
                # RNN aggregation
                old_shape = x.shape
                x = x.contiguous().view(-1, 1, old_shape[-1])
                forward_x, prior_state = self.forward_rnn(x, prior_state)
                all_x.append(x.cpu())
                all_forward_x.append(forward_x.cpu())
                all_prior_result.append(self.prior_fc_out(forward_x).view(old_shape[0], 1, old_shape[1], self.num_edges).cpu())
            reverse_state = None
            all_encoder_result = []
            for timestep in range(num_timesteps-1, -1, -1):
                x = all_x[timestep].cuda()
                reverse_x, reverse_state = self.reverse_rnn(x, reverse_state)
                forward_x = all_forward_x[timestep].cuda()
                
                #x: [batch_size*num_edges, num_timesteps, hidden_size]
                combined_x = torch.cat([forward_x, reverse_x], dim=-1)
                all_encoder_result.append(self.encoder_fc_out(combined_x).view(inputs.size(0), 1, -1, self.num_edges))
            prior_result = torch.cat(all_prior_result, dim=1).cuda(non_blocking=True)
            encoder_result = torch.cat(all_encoder_result, dim=1).cuda(non_blocking=True)
            return prior_result, encoder_result, prior_state

    def single_step_forward(self, inputs, prior_state):
        # Inputs is shape [batch_size, num_vars, input_size]
        x = self.mlp1(inputs)  # 2-layer ELU net per node
        x = self.node2edge(x)
        x = self.mlp2(x)
        x_skip = x
        x = self.edge2node(x)
        x = self.mlp3(x)
        x = self.node2edge(x)
        x = torch.cat((x, x_skip), dim=-1)  # Skip connection
        x = self.mlp4(x)

        old_shape = x.shape
        x  = x.contiguous().view(-1, 1, old_shape[-1])
        old_prior_shape = prior_state[0].shape
        prior_state = (prior_state[0].view(1, old_prior_shape[0]*old_prior_shape[1], old_prior_shape[2]),
                       prior_state[1].view(1, old_prior_shape[0]*old_prior_shape[1], old_prior_shape[2]))

        x, prior_state = self.forward_rnn(x, prior_state)
        prior_result = self.prior_fc_out(x).view(old_shape[0], old_shape[1], self.num_edges)
        prior_state = (prior_state[0].view(old_prior_shape), prior_state[1].view(old_prior_shape))
        return prior_result, prior_state


class DNRI_Decoder(nn.Module):
    def __init__(self, params):
        super(DNRI_Decoder, self).__init__()
        self.num_vars = num_vars =  params['num_vars']
        self.model_type = params['model_type']
        self.layer_num_edge_types = params['layer_num_edge_types']
        self.num_edge_types = params['num_edge_types']
        input_size = params['input_size']
        self.gpu = params['gpu']
        n_hid = params['decoder_hidden']
        edge_types = params['num_edge_types']
        skip_first = params['skip_first']
        out_size = params['input_size']
        do_prob = params['decoder_dropout']

        # self.msg_fc2[k] ○ self.msg_fc1[k] = \tilde{f}^{k}_{e} where ○ denotes function composition
        # see equation (13) in NRI paper
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2*n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(input_size, n_hid, bias=True)
        self.input_i = nn.Linear(input_size, n_hid, bias=True)
        self.input_n = nn.Linear(input_size, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, out_size)

        print('Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob

        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges)), requires_grad=False)

    def get_initial_hidden(self, inputs):
        '''
        get_initial_hidden gets the initial hidden states \tilde{h}^{0}_{j} for all nodes j of the GRU 
        described in equation (15) in the NRI paper.
        '''
        return torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape, device=inputs.device)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, hidden, edges):
        # inputs size: [batch_size, num_vars, input_size] (=[4, 31, 6])
        # hidden size: [batch_size, num_vars, rnn_hidden] (=[4, 31, 256]) where rnn_hidden = params['decoder_hidden']
        # hidden consists of [\tilde{h}^{t}_{1},...,\tilde{h}^{t}_{num_vars}] for each sample in the batch, 
        # see equation (13) in NRI paper
        # edges size: [batch_size, num_edges, num_edge_types] (=[4, 930, 4]) where num_edges = num_vars**2-num_vars
        # 'edges' contains the sampled edge type vector for each edge.
        # Values in parentheses correspond to default command line arguments for dfNRI 4 edge found in run_motion_35.sh.
        if self.training:
            dropout_prob = self.dropout_prob
        else:
            dropout_prob = 0.
        
        # node2edge, see equation (13) in NRI paper
        # self.recv_edges.shape = (num_edges,)
        # receivers.shape = [batch_size, num_edges, rnn_hidden]
        receivers = hidden[:, self.recv_edges, :]
        # self.send_edges.shape = (num_edges,)
        # senders.shape = [batch_size, num_edges, rnn_hidden]
        senders = hidden[:, self.send_edges, :]

        # pre_msg: [batch_size, num_edges, 2*self.msg_out_shape] where self.msg_out_shape = rnn_hidden
        pre_msg = torch.cat([receivers, senders], dim=-1)
        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape, device=inputs.device)
        if self.skip_first_edge_type:
            if self.model_type == 'dnri':
                not_skipped_edge_types = range(1, self.num_edge_types)
            else:
                # In every layer graph a, the first edge type that represents no interaction is skipped.
                # See fNRI paper.
                layer_num_edge_types = self.layer_num_edge_types
                n = len(layer_num_edge_types)
                skipped_edge_types = [sum(layer_num_edge_types[:i]) for i in range(n)]
                not_skipped_edge_types = list(set(range(self.num_edge_types))-set(skipped_edge_types))
                not_skipped_edge_types.sort()
        else:
            not_skipped_edge_types = range(self.num_edge_types)
        norm = len(not_skipped_edge_types)
        # Run separate MLP for every edge type
        # NOTE: to exclude one edge type, simply offset range by 1
        # i corresponds to k in equation (13) in NRI paper
        for i in not_skipped_edge_types:
            msg = torch.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=dropout_prob)
            # 'msg' below contains
            # \tilde{f}^{k}_{e}([\tilde{h}^{t}_{i}, \tilde{h}^{t}_{j}]) of eq. (13) in NRI paper
            # ∀ 1<=i,j<=N ∧ i≠j, i.e. for all edges.
            # msg.shape = [batch_size, num_edges, rnn_hidden]
            msg = torch.tanh(self.msg_fc2[i](msg))
            # 'msg' below contains 
            # z_{ij,k} * \tilde{f}^{k}_{e}([\tilde{h}^{t}_{i}, \tilde{h}^{t}_{j}]) of eq. (13) in NRI paper
            # ∀ edges (i, j).
            # edges[:, :, i:i+1].shape = [batch_size, num_edges, 1]
            msg = msg * edges[:, :, i:i+1]
            # all_msgs = \tilde{h}^{t}_{(i,j)} for all edges (i,j), see eq. (13) of NRI paper
            all_msgs += msg/norm
        
        # This step sums all of the messages per node, see eq. (14) in the NRI paper.
        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous() / (self.num_vars - 1) # Average

        # GRU-style gated aggregation, see equation (15) in NRI paper
        inp_r = self.input_r(inputs).view(inputs.size(0), self.num_vars, -1)
        inp_i = self.input_i(inputs).view(inputs.size(0), self.num_vars, -1)
        inp_n = self.input_n(inputs).view(inputs.size(0), self.num_vars, -1)
        r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs))
        i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs))
        n = torch.tanh(inp_n + r*self.hidden_h(agg_msgs))
        hidden = (1 - i)*n + i*hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=dropout_prob)
        pred = self.out_fc3(pred)

        # See equation (16) in the NRI paper.
        pred = inputs + pred

        return pred, hidden


class DNRI_MLP_Decoder(nn.Module):
    def __init__(self, params):
        super(DNRI_MLP_Decoder, self).__init__()
        num_vars = params['num_vars']
        edge_types = params['num_edge_types']
        n_hid = params['decoder_hidden']
        msg_hid = params['decoder_hidden']
        msg_out = msg_hid #TODO: make this a param
        skip_first = params['skip_first']
        n_in_node = params['input_size']

        do_prob = params['decoder_dropout']
        in_size = n_in_node
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * in_size, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        out_size = n_in_node
        self.out_fc1 = nn.Linear(in_size + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, out_size)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob
        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges)), requires_grad=False)

    def get_initial_hidden(self, inputs):
        return None

    def forward(self, inputs, hidden, edges):

        # single_timestep_inputs has shape
        # [batch_size, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]
        # Node2edge
        receivers = inputs[:, self.recv_edges, :]
        senders = inputs[:, self.send_edges, :]
        pre_msg = torch.cat([receivers, senders], dim=-1)

        if inputs.is_cuda:
            all_msgs = torch.cuda.FloatTensor(pre_msg.size(0), pre_msg.size(1),
                                self.msg_out_shape).fill_(0.)
        else:
            all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                self.msg_out_shape)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0
        if self.training:
            p = self.dropout_prob
        else:
            p = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=p)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * edges[:, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=p)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=p)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return inputs + pred, None