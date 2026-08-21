from __future__ import annotations

from typing import Any


def topk_activations(values: Any, k: int) -> Any:
    import torch

    k = min(int(k), values.shape[-1])
    top_values, top_indices = torch.topk(values, k=k, dim=-1)
    return torch.zeros_like(values).scatter(-1, top_indices, top_values)


class TopKSAE:
    @staticmethod
    def build(d_model: int, n_features: int, k: int, *, relu: bool = True) -> Any:
        import torch

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = torch.nn.Linear(d_model, n_features)
                self.decoder = torch.nn.Linear(n_features, d_model, bias=True)
                self.k = int(k)
                self.use_relu = bool(relu)
                torch.nn.init.kaiming_uniform_(self.encoder.weight)
                with torch.no_grad():
                    self.decoder.weight.copy_(self.encoder.weight.T)
                    self.normalize_decoder()

            def normalize_decoder(self) -> None:
                self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))

            def encode(self, inputs: Any) -> Any:
                preactivations = self.encoder(inputs)
                if self.use_relu:
                    preactivations = torch.relu(preactivations)
                return topk_activations(preactivations, self.k)

            def decode(self, features: Any) -> Any:
                return self.decoder(features)

            def forward(self, inputs: Any) -> tuple[Any, Any]:
                features = self.encode(inputs)
                return self.decode(features), features

        return Module()


def load_sae(path: str, device: Any = "cpu") -> tuple[Any, dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload["config"]
    model = TopKSAE.build(config["d_model"], config["n_features"], config["k"], relu=config.get("relu", True))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload
