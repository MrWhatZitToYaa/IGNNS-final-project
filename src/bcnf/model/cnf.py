from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn

from bcnf.model.feature_network import FeatureNetwork


class ConditionalInvertibleLayer(nn.Module):
    log_det_J: float | torch.Tensor | None
    n_conditions: int

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass


class ConditionalNestedNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_conditions: int, dropout: float = 0.2, device: str = "cpu") -> None:
        super(ConditionalNestedNeuralNetwork, self).__init__()

        self.n_conditions = n_conditions

        self.layers = nn.Sequential(
            nn.Linear(input_size + n_conditions, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size * 2)
        ).to(device)

    def to(self, device: str) -> "ConditionalNestedNeuralNetwork":  # type: ignore
        super().to(device)
        self.device = device
        self.layers.to(device)

        return self

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.n_conditions > 0:
            # Concatenate the input with the condition
            x = torch.cat([x, y], dim=1)

        # Get the translation coefficients t and the scale coefficients s from the neural network
        t, s = self.layers(x).chunk(2, dim=1)

        # Return the coefficients
        return t, torch.tanh(s)


class ConditionalAffineCouplingLayer(ConditionalInvertibleLayer):
    def __init__(self, input_size: int, hidden_size: int, n_conditions: int, dropout: float = 0.2, device: str = "cpu") -> None:
        super(ConditionalAffineCouplingLayer, self).__init__()

        self.n_conditions = n_conditions
        self.log_det_J: torch.Tensor = torch.zeros(1).to(device)

        # Create the nested neural network
        self.nn = ConditionalNestedNeuralNetwork(
            input_size=np.ceil(input_size / 2).astype(int),
            output_size=np.floor(input_size / 2).astype(int),
            hidden_size=hidden_size,
            n_conditions=n_conditions,
            dropout=dropout,
            device=device)

    def to(self, device: str) -> "ConditionalAffineCouplingLayer":  # type: ignore
        super().to(device)
        self.device = device
        self.log_det_J = self.log_det_J.to(device)
        self.nn.to(device)

        return self

    def forward(self, x: torch.Tensor, y: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        # Split the input into two halves
        x_a, x_b = x.chunk(2, dim=1)

        # Get the coefficients from the neural network
        t, log_s = self.nn.forward(x_a, y)

        # Apply the transformation
        z_a = x_a  # skip connection
        z_b = torch.exp(log_s) * x_b + t  # affine transformation

        # Calculate the log determinant of the Jacobian
        if log_det_J:
            self.log_det_J = log_s.sum(dim=1)

        # Return the output
        return torch.cat([z_a, z_b], dim=1)

    def inverse(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Split the input into two halves
        z_a, z_b = z.chunk(2, dim=1)

        # Get the coefficients from the neural network
        t, log_s = self.nn.forward(z_a, y)

        # Apply the inverse transformation
        x_a = z_a
        x_b = (z_b - t) * torch.exp(- log_s)

        # Return the output
        return torch.cat([x_a, x_b], dim=1)


class OrthonormalTransformation(ConditionalInvertibleLayer):
    def __init__(self, input_size: int) -> None:
        super(OrthonormalTransformation, self).__init__()

        self.log_det_J: float = 0

        # Create the random orthonormal matrix via QR decomposition
        self.orthonormal_matrix: torch.Tensor = torch.linalg.qr(torch.randn(input_size, input_size))[0]
        self.orthonormal_matrix.requires_grad = False

    def to(self, device: str) -> "OrthonormalTransformation":  # type: ignore
        super().to(device)
        self.device = device
        self.orthonormal_matrix = self.orthonormal_matrix.to(device)

        return self

    def forward(self, x: torch.Tensor, y: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        # Apply the transformation
        return x @ self.orthonormal_matrix

    def inverse(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Apply the inverse transformation
        return z @ self.orthonormal_matrix.T


class CondRealNVP(ConditionalInvertibleLayer):
    def __init__(self, input_size: int, hidden_size: int, n_blocks: int, n_conditions: int, feature_network: FeatureNetwork | None, dropout: float = 0.2, device: str = "cpu"):
        super(CondRealNVP, self).__init__()

        if n_conditions == 0 or feature_network is None:
            self.h = nn.Identity()
        else:
            self.h = feature_network

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.n_conditions = n_conditions
        self.device = device
        self.dropout = dropout
        self.log_det_J: torch.Tensor = torch.zeros(1).to(self.device)

        # Create the network
        self.layers: list[ConditionalInvertibleLayer] = []
        for _ in range(self.n_blocks - 1):
            self.layers.append(ConditionalAffineCouplingLayer(self.input_size, self.hidden_size, self.n_conditions, dropout=self.dropout, device=self.device))
            self.layers.append(OrthonormalTransformation(self.input_size))

        # Add the final affine coupling layer
        self.layers.append(ConditionalAffineCouplingLayer(self.input_size, self.hidden_size, self.n_conditions, dropout=self.dropout, device=self.device))

    def to(self, device: str) -> "CondRealNVP":  # type: ignore
        super().to(device)
        self.device = device
        self.log_det_J = self.log_det_J.to(device)
        for layer in self.layers:
            layer.to(device)

        return self

    def forward(self, x: torch.Tensor, y: torch.Tensor, log_det_J: bool = False) -> torch.Tensor:
        # Apply the feature network to y
        y = self.h(y)

        # Apply the network

        if log_det_J:
            self.log_det_J = torch.zeros(x.shape[0]).to(self.device)

            for layer in self.layers:
                x = layer(x, y, log_det_J)
                self.log_det_J += layer.log_det_J

            return x

        for layer in self.layers:
            x = layer(x, y, log_det_J)

        return x

    def inverse(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Apply the feature network to y
        y = self.h(y)

        # Apply the network in reverse
        for layer in reversed(self.layers):
            z = layer.inverse(z, y)

        return z

    def sample(self, n_samples: int, y: torch.Tensor, sigma: float = 1, outer: bool = False, verbose: bool = False) -> torch.Tensor:
        """
        Sample from the model.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        y : torch.Tensor
            The conditions used for sampling.
            If 1st order tensor and len(y) == n_input_conditions, the same conditions are used for all samples.
            If 2nd order tensor, y.shape must be (n_samples, n_input_conditions), and each row is used as the conditions for each sample.
        sigma : float
            The standard deviation of the normal distribution to sample from.
        outer : bool
            If True, the conditions are broadcasted to match the shape of the samples.
            If False, the conditions are matched to the shape of the samples.
        verbose : bool
            If True, print debug information.

        Returns
        -------
        torch.Tensor
            The generated samples.
        """

        print(f'{y.shape=}')

        y = y.to(self.device)

        if y.ndim == 1:
            if verbose:
                print('Broadcasting')
            # if len(y) != n_input_conditions:
            #     raise ValueError(f"y must have length {n_input_conditions}, but got len(y) = {len(y)}")

            # Generate n_samples for each condition in y
            z = sigma * torch.randn(n_samples, self.input_size).to(self.device)
            y = y.repeat(n_samples, 1)

            # Apply the inverse network
            return self.inverse(z, y).view(n_samples, self.input_size)
        elif y.ndim == 2:
            if outer:
                if verbose:
                    print('Outer')
                # if y.shape[1] != n_input_conditions:
                #     raise ValueError(f"y must have shape (n_samples_per_condition, {n_input_conditions}), but got y.shape = {y.shape}")

                n_samples_per_condition = y.shape[0]

                # Generate n_samples for each condition in y
                z = sigma * torch.randn(n_samples * n_samples_per_condition, self.input_size).to(self.device)
                y = y.repeat(n_samples, 1)

                # Apply the inverse network
                return self.inverse(z, y).view(n_samples, n_samples_per_condition, self.input_size)
            else:
                if verbose:
                    print('Matching')
                # if y.shape[0] != n_samples or y.shape[1] != n_input_conditions:
                #     raise ValueError(f"y must have shape (n_samples, {n_input_conditions}), but got y.shape = {y.shape}")

                # Use y_i as the condition for the i-th sample
                z = sigma * torch.randn(n_samples, self.input_size).to(self.device)

                # Apply the inverse network
                return self.inverse(z, y).view(n_samples, self.input_size)
        else:
            raise ValueError(f"y must be a 1st or 2nd order tensor, but got y.shape = {y.shape}")