import torch
import numpy as np


class KalmanFilter:
    def __init__(self, transfer_matrix=None, observation_matrix=None, transfer_covariance=None, observation_covariance=None,
                 requires_grad=True):
        self.transfer_matrix = transfer_matrix.clone().detach()
        self.observation_matrix = observation_matrix.clone().detach()
        self.transfer_covariance_p1log = torch.log(
            transfer_covariance + 1).requires_grad_(requires_grad)
        self.observation_covariance_p1log = torch.log(
            observation_covariance + 1).requires_grad_(requires_grad)
        self.state_dim = self.transfer_matrix.shape[0]
        self.obs_dim = self.observation_matrix.shape[0]

    def filter(self, Y, initial_state_mean=None, initial_state_cov=None):
        N = len(Y)
        self.predicted_state_mean = [torch.zeros(
            (self.state_dim)).double() for _ in range(N)]
        self.predicted_state_cov = [torch.zeros(
            (self.state_dim, self.state_dim)).double() for _ in range(N)]
        self.filtered_state_mean = [torch.zeros(
            (self.state_dim)).double() for _ in range(N)]
        self.filtered_state_cov = [torch.zeros(
            (self.state_dim, self.state_dim)).double() for _ in range(N)]
        self.predicted_observation_mean = [torch.zeros(
            (self.obs_dim)).double() for _ in range(N)]
        self.predicted_observation_cov = [torch.zeros(
            (self.obs_dim, self.obs_dim)).double() for _ in range(N)]
        self.predicted_error = [torch.zeros(
            (self.obs_dim)).double() for _ in range(N)]
        self.kalman_gain = [torch.zeros(
            (self.state_dim, self.obs_dim)) for _ in range(N)]

        self.transfer_covariance = torch.exp(
            self.transfer_covariance_p1log) - 1
        self.observation_covariance = torch.exp(
            self.observation_covariance_p1log) - 1

        self.initial_state_mean = initial_state_mean if initial_state_mean is not None else torch.zeros(
            (self.state_dim)).double()
        self.initial_state_cov = initial_state_cov if initial_state_cov is not None else (
            torch.eye(self.state_dim) * 1e7).double()
        self.loglikelihoods = torch.zeros(N).double()

        for i in range(N):
            # predict state
            if i == 0:
                self.predicted_state_mean[i] = self.transfer_matrix @ self.initial_state_mean
                self.predicted_state_cov[i] = (self.transfer_matrix @ self.initial_state_cov @ self.transfer_matrix.T
                                               + self.transfer_covariance)
            else:
                self.predicted_state_mean[i] = self.transfer_matrix @ self.filtered_state_mean[i-1]
                self.predicted_state_cov[i] = (self.transfer_matrix @ self.filtered_state_cov[i-1] @ self.transfer_matrix.T
                                               + self.transfer_covariance)
            # predict observation
            self.predicted_observation_mean[i] = self.observation_matrix @ self.predicted_state_mean[i]
            self.predicted_observation_cov[i] = (self.observation_matrix @ self.predicted_state_cov[i] @ self.observation_matrix.T
                                                 + self.observation_covariance)
            self.predicted_error[i] = Y[i] - self.predicted_observation_mean[i]

            # filter state
            self.kalman_gain[i] = (self.predicted_state_cov[i] @ self.observation_matrix.T
                                   @ torch.inverse(self.predicted_observation_cov[i]))
            self.filtered_state_mean[i] = (self.predicted_state_mean[i]
                                           + torch.squeeze(self.kalman_gain[i] @ self.predicted_error[i].view(self.obs_dim, -1)))
            self.filtered_state_cov[i] = (torch.eye(self.state_dim).double()
                                          - self.kalman_gain[i] @ self.observation_matrix) @ self.predicted_state_cov[i]
            self.loglikelihoods[i] = -0.5 * (torch.log(torch.det(self.predicted_observation_cov[i]))
                                             + self.predicted_error[i] @ torch.inverse(
                                                 self.predicted_observation_cov[i]) @ self.predicted_error[i]
                                             + self.obs_dim * torch.log(2 * torch.tensor([np.pi]).double()))

        self.negloglikelihood = -1 * self.loglikelihoods.sum()
        return self

    def smooth(self, Y, initial_state_mean=None, initial_state_cov=None):
        self.filter(Y, initial_state_mean=None, initial_state_cov=None)
        N = len(Y)
        self.smoothed_state_mean = [torch.zeros(
            (self.state_dim)).double() for _ in range(N)]
        self.smoothed_state_cov = [torch.zeros(
            (self.state_dim, self.state_dim)).double() for _ in range(N)]

        for i in range(N):
            j = N-i-1
            if i == 0:
                self.smoothed_state_mean[j] = self.filtered_state_mean[j]
                self.smoothed_state_cov[j] = self.filtered_state_cov[j]
            else:
                smooth_gain = self.filtered_state_cov[j] @ self.transfer_matrix.T @ torch.inverse(
                    self.predicted_state_cov[j+1])
                self.smoothed_state_mean[j] = (self.filtered_state_mean[j]
                                               + smooth_gain @ (self.smoothed_state_mean[j+1] - self.predicted_state_mean[j+1]))
                self.smoothed_state_cov[j] = (self.filtered_state_cov[j]
                                              + smooth_gain @ (self.smoothed_state_cov[j+1] - self.predicted_state_cov[j+1])
                                              @ smooth_gain.T)
        return self

    def predict(self, n_ahead):
        predicted_state_mean = [torch.zeros(
            (self.state_dim)).double() for _ in range(n_ahead)]
        predicted_state_cov = [torch.zeros(
            (self.state_dim, self.state_dim)).double() for _ in range(n_ahead)]
        predicted_obs_mean = [torch.zeros(
            (self.obs_dim)).double() for _ in range(n_ahead)]
        predicted_obs_cov = [torch.zeros(
            (self.obs_dim, self.obs_dim)).double() for _ in range(n_ahead)]
        for i in range(n_ahead):
            if i == 0:
                predicted_state_mean[i] = self.transfer_matrix @ self.predicted_state_mean[-1]
                predicted_state_cov[i] = (self.transfer_matrix @ self.predicted_state_cov[-1] @ self.transfer_matrix.T
                                          + self.transfer_covariance)
            else:
                predicted_state_mean[i] = self.transfer_matrix @ predicted_state_mean[i-1]
                predicted_state_cov[i] = (self.transfer_matrix @ predicted_state_cov[i-1] @ self.transfer_matrix.T
                                          + self.transfer_covariance)
            predicted_obs_mean[i] = self.observation_matrix @ predicted_state_mean[i]
            predicted_obs_cov[i] = (self.observation_matrix @ predicted_state_cov[i] @ self.observation_matrix.T
                                    + self.observation_covariance)
        return {"state_mean": predicted_state_mean, "state_cov": predicted_state_cov,
                "obs_mean": predicted_obs_mean, "obs_cov": predicted_obs_cov}

    def fit(self, Y, initial_state_mean=None, initial_state_cov=None, learning_rate=1e-3, n_epoch=10,
            learning_mask_transfer_cov=None, learning_mask_observation_cov=None):
        if learning_mask_transfer_cov is None:
            lerning_mask_transfer_cov = torch.ones_like(
                self.transfer_covariance_p1log)
        if learning_mask_observation_cov is None:
            learning_mask_observation_cov = torch.ones_like(
                self.observation_covariance_p1log)

        self.loss_history = [0] * n_epoch
        for i in range(n_epoch):
            if i != 0:
                self.transfer_covariance_p1log.grad.data.zero_()
                self.observation_covariance_p1log.grad.data.zero_()
            self.filter(Y)
            self.loss_history[i] = self.negloglikelihood.item()
            self.negloglikelihood.backward()

            self.transfer_covariance_p1log.grad.data[self.transfer_covariance.data < 0] = -1e10
            self.observation_covariance_p1log.grad.data[self.observation_covariance.data < 0] = -1e10

            self.transfer_covariance_p1log.data.sub_(self.transfer_covariance_p1log.grad.data
                                                     * learning_rate * learning_mask_transfer_cov)
            self.observation_covariance_p1log.data.sub_(self.observation_covariance_p1log.grad.data
                                                        * learning_rate * learning_mask_observation_cov)
        return self


class ParticleFilter:
    def __init__(self, transfer_matrix=None, observation_matrix=None, transfer_covariance=None, observation_covariance=None):
        self.transfer_matrix = transfer_matrix.clone().detach()
        self.observation_matrix = observation_matrix.clone().detach()
        self.transfer_covariance_p1log = torch.log(
            transfer_covariance + 1).requires_grad_(True)
        self.observation_covariance_p1log = torch.log(
            observation_covariance + 1).requires_grad_(True)
        self.state_dim = self.transfer_matrix.shape[0]
        self.obs_dim = self.observation_matrix.shape[0]

    def _sys_resample_idx(self, n_sample, probs):
        sampled_cumlative_probs = (
            (torch.arange(1, n_sample + 1).double() -
             torch.rand(1).double()) / n_sample
        ).unsqueeze(-1)
        cumlative_probs = torch.cumsum(
            torch.cat(
                (torch.tensor([0]).double(), probs)
            ), dim=0
        )
        resampled_idx = (cumlative_probs <
                         sampled_cumlative_probs).sum(axis=1) - 1
        return resampled_idx

    def filter(self, Y, n_particle=1000, initial_state_mean=None, initial_state_cov=None):
        N = len(Y)
        self.n_particle = n_particle
        self.predicted_states = [torch.zeros(
            (self.n_particle, self.state_dim)) for _ in range(N)]
        self.log_weights = [torch.log(torch.ones(
            (self.n_particle)).double()/n_particle) for _ in range(N)]
        self.resample_idx = [torch.zeros((self.n_particle)) for _ in range(N)]
        self.predicted_states_resampled = [torch.zeros(
            (self.n_particle, self.state_dim)) for _ in range(N)]
        self.effective_sample_size = torch.zeros((N,))

        self.transfer_covariance = torch.exp(
            self.transfer_covariance_p1log) - 1
        self.observation_covariance = torch.exp(
            self.observation_covariance_p1log) - 1

        self.initial_state_mean = initial_state_mean if initial_state_mean is not None else torch.zeros(
            (self.state_dim)).double()
        self.initial_state_cov = initial_state_cov if initial_state_cov is not None else (
            torch.eye(self.state_dim) * 1e3).double()
        # self.loglikelihoods = torch.zeros(N).double()

        for i in range(N):
            # predict state
            if i == 0:
                self.normal_dist_state = torch.distributions.multivariate_normal.MultivariateNormal(
                    self.transfer_matrix @ self.initial_state_mean, self.initial_state_cov
                )
                self.predicted_states[i] = self.normal_dist_state.sample(
                    (n_particle,))
            else:
                self.normal_dist_state = torch.distributions.multivariate_normal.MultivariateNormal(
                    (self.transfer_matrix @
                     self.predicted_states_resampled[i-1].T).T, self.transfer_covariance
                )
                self.predicted_states[i] = self.normal_dist_state.sample(
                    (1,)).squeeze(0)

            # filter state
            self.normal_dist_observation = torch.distributions.multivariate_normal.MultivariateNormal(
                (self.observation_matrix @
                 self.predicted_states[i].T).T, self.observation_covariance
            )
            self.log_prob = self.normal_dist_observation.log_prob(Y[i].T)
            self.log_weights[i] = self.log_weights[i-1] + self.log_prob
            self.log_weights[i] = self.log_weights[i] - \
                torch.log(sum(torch.exp(self.log_weights[i])))

            # resampling
            # ~~normal resampling~~
            # self.resampler = torch.distributions.categorical.Categorical(torch.exp(self.log_weights[i]))
            # self.resampled_idx = self.resampler.sample((n_particle,))
            # ~~~~~~~~~~~~~~~~~
            # ~~system resampling~~
            self.resample_idx[i] = self._sys_resample_idx(
                n_particle, torch.exp(self.log_weights[i]))
            self.predicted_states_resampled[i] = self.predicted_states[i][self.resample_idx[i]]

            # ~~~~~~~~~~~~~~~~~
            self.effective_sample_size[i] = 1 / \
                sum(torch.exp(self.log_weights[i])**2)
            self.log_weights[i] = torch.log(torch.ones(
                (self.n_particle)).double()/n_particle)

        return self

    def predict(self, n_ahead=10):
        predicted_states = [torch.zeros(
            (self.n_particle, self.state_dim)).double() for _ in range(n_ahead)]
        predicted_obs = [torch.zeros(
            (self.n_particle, self.obs_dim)).double() for _ in range(n_ahead)]
        for i in range(n_ahead):
            # predict state
            if i == 0:
                normal_dist_state = torch.distributions.multivariate_normal.MultivariateNormal(
                    (self.transfer_matrix @
                     self.predicted_states[-1].T).T, self.transfer_covariance
                )
            else:
                normal_dist_state = torch.distributions.multivariate_normal.MultivariateNormal(
                    (self.transfer_matrix @
                     predicted_states[i-1].T).T, self.transfer_covariance
                )
            predicted_states[i] = normal_dist_state.sample((1,)).squeeze(0)

            # predict observation
            normal_dist_observation = torch.distributions.multivariate_normal.MultivariateNormal(
                (self.observation_matrix @
                 predicted_states[i].T).T, self.observation_covariance
            )
            predicted_obs[i] = normal_dist_observation.sample((1,)).squeeze(0)
        return predicted_states, predicted_obs

    def smooth(self, smooth_length=None):
        N = len(self.predicted_states)
        self.smoothed_states = [torch.zeros(
            (self.n_particle, self.state_dim)) for _ in range(N)]
        for i in range(N):
            idx = torch.arange(self.n_particle)
            smooth_end = N if smooth_length is None else i + smooth_length + 1
            smooth_end = min(smooth_end, N)
            for j in range(i+1, smooth_end):
                idx = idx[self.resample_idx[j]]
            self.smoothed_states[i] = self.predicted_states_resampled[i][idx]
