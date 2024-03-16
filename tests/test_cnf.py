import unittest

import torch

from bcnf.models.cnf import ConditionalAffineCouplingLayer, CondRealNVP


class TestCNFInvertibility(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cpu"
        self.input_size = 7
        self.hidden_size = 19
        self.n_conditions = 5

        self.x = torch.randn(17, 7)
        self.y = torch.randn(17, 5)

    def test_conditional_affine_coupling_layer(self) -> None:
        coupling_layer = ConditionalAffineCouplingLayer(
            input_size=self.input_size,
            nested_sizes=[self.hidden_size] * 5,
            n_conditions=self.n_conditions
        )
        coupling_layer.eval()

        z = coupling_layer.forward(self.x, self.y)

        self.assertEqual(z.shape, self.x.shape)

        x_hat = coupling_layer.inverse(z, self.y)

        self.assertTrue(torch.allclose(x_hat, self.x, atol=1e-5))

    def test_conditional_real_nvp(self) -> None:
        real_nvp = CondRealNVP(
            size=self.input_size,
            nested_sizes=[self.hidden_size] * 5,
            n_blocks=5,
            n_conditions=self.n_conditions,
            feature_network=None
        )
        real_nvp.eval()

        z = real_nvp.forward(self.x, self.y)

        self.assertEqual(z.shape, self.x.shape)

        x_hat = real_nvp.inverse(z, self.y)

        self.assertTrue(torch.allclose(x_hat, self.x, atol=1e-5))