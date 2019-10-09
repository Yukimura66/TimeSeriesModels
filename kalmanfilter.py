import torch
import numpy as np


class KalmanFilter:
    def __init__(self, transfer_matrix=None, observation_matrix=None,
                 transfer_covariance=None, observation_covariance=None):
        self.transfer_matrix = transfer_matrix.clone().detach()
        self.observation_matrix = observation_matrix.clone().detach()
        self.transfer_covariance_raw = (transfer_covariance.clone().detach()
                                        .requires_grad_(True))
        self.observation_covariance_raw = observation_covariance.clone(
        ).detach().requires_grad_(True)
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
        self.transfer_covariance = torch.exp(self.transfer_covariance_raw)
        self.transfer_covariance.data.sub_(
            torch.tensor([[0, 1], [1, 0]]).double())
        self.observation_covariance = torch.exp(
            self.observation_covariance_raw)

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
                self.predicted_error[i] = Y[i] - \
                    self.predicted_observation_mean[i]

                # filter state
                kalman_gain = self.predicted_state_cov[i] @ self.observation_matrix.T @ torch.inverse(
                    self.predicted_observation_cov[i])
                self.filtered_state_mean[i] = (self.predicted_state_mean[i]
                                               + torch.squeeze(kalman_gain @ self.predicted_error[i].view(self.obs_dim, -1)))
                self.filtered_state_cov[i] = (torch.eye(self.state_dim).double()
                                              - kalman_gain @ self.observation_matrix) @ self.predicted_state_cov[i]
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
                self.transfer_covariance_raw)
        if learning_mask_observation_cov is None:
            learning_mask_observation_cov = torch.ones_like(
                self.observation_covariance_raw)

        self.loss_history = [0] * n_epoch
        for i in range(n_epoch):
            if i != 0:
                self.transfer_covariance_raw.grad.data.zero_()
                self.observation_covariance_raw.grad.data.zero_()
            self.filter(Y)
            self.loss_history[i] = self.negloglikelihood.item()
            self.negloglikelihood.backward()
            self.transfer_covariance_raw.data.sub_(self.transfer_covariance_raw.grad.data
                                                   * learning_rate * learning_mask_transfer_cov)
            self.observation_covariance_raw.data.sub_(self.observation_covariance_raw.grad.data
                                                      * learning_rate * learning_mask_observation_cov)
        return self
