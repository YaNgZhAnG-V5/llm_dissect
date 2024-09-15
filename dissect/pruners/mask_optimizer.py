from typing import List

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from alive_progress import alive_it
from torch import nn
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from dissect.models import L0Mask
from dissect.utils import suppress_output, suppress_tqdm


class MaskOptimizer:
    def __init__(
        self,
        model: nn.Module,
        target_modules: List[str],
        outputs: torch.Tensor,
        data_loader: DataLoader,
        distance_metric: str,
        num_iterations: int,
        alpha: float,
        beta: float,
        gamma: float,
        eta: float,
        lr_mask: float,
        lr_mu: float,
        device: torch.device,
        logger,
        lamb_init: str = "random",
        target_sparsity: float = 0.25,
        use_lagrangian: bool = False,
        use_lagrangian_proxy: bool = True,
        gradient_checkpointing: bool = True,
        warm_up: int = 100,
    ):
        """
        Initializes the MaskOptimizer with required parameters.

        Parameters:
        -----------
        - model: The neural network model to prune.
        - target_modules: List of module names to apply masks.
        - outputs: Original model outputs for comparison.
        - data_loader: DataLoader for pruning dataset.
        - distance_metric: Metric to measure output distance.
        - num_iterations: Number of optimization iterations.
        - alpha: Weight for sparsity regularization.
        - beta: Weight for polar regularization.
        - gamma: Weight for Lagrangian's first-order term.
        - eta: Weight for Lagrangian's second-order term.
        - lr_mask: Learning rate for mask optimizer.
        - lr_mu: Learning rate for Lagrangian multipliers optimizer.
        - device: Device to perform computations on.
        - logger: Logger for logging information.
        - lamb_init: Initialization strategy for lambdas.
        - target_sparsity: Desired sparsity level (0-1).
        - use_lagrangian: Flag to use Lagrangian multipliers.
        - use_lagrangian: Flag to use the loss for Lagrangian optimizaiotn but fixing the multipliers.
        - gradient_checkpointing: Flag to enable gradient checkpointing.
        - warm_up: warm_up iterations before using regularization
        """
        self.model = model
        self.target_modules = target_modules
        self.outputs = outputs
        self.data_loader = data_loader
        self.distance_metric = distance_metric
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.lr_mask = lr_mask
        self.lr_mu = lr_mu
        self.device = device
        self.logger = logger
        self.lamb_init = lamb_init
        self.target_sparsity = target_sparsity
        self.use_lagrangian = use_lagrangian
        self.use_lagrangian_proxy = use_lagrangian_proxy
        self.gradient_checkpointing = gradient_checkpointing
        self.warm_up = warm_up

        self.lambs = []
        self._initialize_masks()

        params_to_optimize = []
        for lamb in self.lambs:
            lamb.train()
            params_to_optimize.extend(list(lamb.parameters()))
        self.optimizer_mask = torch.optim.Adam(params_to_optimize, lr=self.lr_mask)

        if self.use_lagrangian:
            # Initialize Lagrangian multipliers as Parameters for optimization
            self.mu = nn.Parameter(torch.zeros(1, device=self.device))
            self.nu = nn.Parameter(torch.zeros(1, device=self.device))
            self.optimizer_mu = torch.optim.Adam([self.mu, self.nu], lr=self.lr_mu)
        else:
            self.mu = None
            self.nu = None
            self.optimizer_mu = None
        if self.gradient_checkpointing:
            # Setup Accelerator for mixed precision and gradient checkpointing
            self.accelerator = Accelerator(mixed_precision="fp16")
            self.model.gradient_checkpointing_enable()
            # Prepare model, optimizers, and dataloader
            self.model, self.optimizer_mask, self.data_loader = self.accelerator.prepare(
                self.model, self.optimizer_mask, self.data_loader
            )
        if self.use_lagrangian:
            self.optimizer_mu = self.accelerator.prepare(self.optimizer_mu)

    def _initialize_masks(self):
        """
        Initializes the masks (L0Mask instances) and registers forward hooks.
        """
        for name, module in self.model.named_modules():
            for target_module in self.target_modules:
                if target_module in name.split(".")[-1]:
                    lamb = L0Mask(
                        shape=(1,), temperature=2.0 / 3.0, droprate_init=self.target_sparsity, device=self.device
                    )
                    module.register_forward_hook(self._create_concrete_mask_hook(lamb))
                    self.lambs.append(lamb)

    @staticmethod
    def _create_concrete_mask_hook(lamb: L0Mask):
        """
        Creates a forward hook for applying the Concrete mask.

        Parameters:
        -----------
        - lamb: The L0Mask instance.

        Returns:
        --------
        - A hook function.
        """

        def hook(module, input, output):
            return output * lamb().to(output.device)

        return hook

    def _compute_distance(self, original_output: torch.Tensor, output_logits: torch.Tensor) -> torch.Tensor:
        """
        Computes the distance between original and current model outputs based on the selected metric.

        Parameters:
        -----------
        - original_output: Original model output logits.
        - output_logits: Current model output logits.

        Returns:
        --------
        - Computed distance.
        """
        if self.distance_metric == "norm":
            return F.mse_loss(original_output, output_logits)
        elif self.distance_metric == "angular_distance":
            cosine_similarity = cosine_similarity(original_output, output_logits, dim=-1)
            cosine_similarity = torch.clamp(cosine_similarity, 0.0, 1.0)
            angular_distance = torch.acos(cosine_similarity)
            return angular_distance.mean()
        elif self.distance_metric == "kl_divergence":
            return F.kl_div(
                F.log_softmax(output_logits, dim=-1),
                F.softmax(original_output, dim=-1),
                reduction="mean",
            )
        elif self.distance_metric == "js_divergence":
            mean_prob = 0.5 * (F.softmax(original_output, dim=-1) + F.softmax(output_logits, dim=-1))
            return 0.5 * (
                F.kl_div(F.log_softmax(original_output, dim=-1), mean_prob, reduction="mean")
                + F.kl_div(F.log_softmax(output_logits, dim=-1), mean_prob, reduction="mean")
            )
        else:
            raise NotImplementedError(f"Unsupported distance metric: {self.distance_metric}")

    def optimize_masks(self):
        """
        Runs the optimization loop to learn the masks using Lagrangian optimization.
        """
        self.model.train()
        total_layers = len(self.lambs)
        target_total = self.target_sparsity * total_layers

        for it in range(self.num_iterations):
            self.optimizer_mask.zero_grad()
            if self.use_lagrangian:
                self.optimizer_mu.zero_grad()

            mean_distance = []
            mean_loss = []
            # batch_count = 0

            with suppress_output(), suppress_tqdm():
                for data, original_output in alive_it(
                    zip(self.data_loader, self.outputs), total=len(self.data_loader), enrich_print=False, disable=True
                ):
                    original_output = original_output.to(self.device)
                    data = BatchEncoding(data).to(self.device)
                    output = self.model(**data)
                    output_logits = output.logits.to(self.device)

                    distance = self._compute_distance(original_output, output_logits)

                    # Compute L0 norms for all masks
                    l0_tensor = torch.cat([lamb.l0_norm() for lamb in self.lambs])

                    # Constraint: sum(l0_tensor) == target_total
                    constraint = torch.sum(l0_tensor) - target_total

                    if self.use_lagrangian:
                        # Lagrangian Loss: L = similarity_loss + mu * constraint + nu * constraint^2
                        loss = distance + self.mu * constraint + self.nu * (constraint**2)
                    elif self.use_lagrangian_proxy:
                        if it < self.warm_up:
                            loss = distance
                        else:
                            loss = distance + self.gamma * torch.abs(constraint) + self.eta * torch.abs(constraint)
                    else:
                        # Basic regularization without Lagrangian
                        lamb_tensor = torch.sigmoid(torch.cat([lamb.z_loga for lamb in self.lambs]))
                        sparsity_regularization = self.alpha * F.l1_loss(lamb_tensor, torch.zeros_like(lamb_tensor))
                        polar_regularization = self.beta * F.binary_cross_entropy_with_logits(
                            torch.cat([lamb.z_loga for lamb in self.lambs]),
                            torch.zeros_like(torch.cat([lamb.z_loga for lamb in self.lambs])).to(self.device),
                        )
                        loss = distance + sparsity_regularization + polar_regularization

                    # Backpropagate
                    if self.gradient_checkpointing:
                        self.accelerator.backward(loss)
                    else:
                        loss.backward()

                    # Update mask parameters
                    self.optimizer_mask.step()

                    if self.use_lagrangian:
                        # Update Lagrangian multipliers via gradient ascent
                        # Compute the Lagrangian components separately
                        # Loss with respect to multipliers: mu * constraint + nu * constraint^2
                        loss_mu = self.mu * constraint + self.nu * (constraint**2)
                        # To maximize loss_mu, minimize -loss_mu
                        if self.gradient_checkpointing:
                            self.accelerator.backward(-loss_mu)
                        else:
                            loss_mu_neg = -loss_mu
                            loss_mu_neg.backward()
                        self.optimizer_mu.step()

                        # Detach multipliers to prevent backpropagation through them in mask optimizer
                        self.mu.data.clamp_(min=0.0)  # Ensure mu is non-negative
                        self.nu.data.clamp_(min=0.0)  # Ensure nu is non-negative

                    mean_distance.append(distance.item())
                    mean_loss.append(loss.item())
            mean_distance = sum(mean_distance) / len(mean_distance)
            mean_loss = sum(mean_loss) / len(mean_loss)
            # batch_count += 1

            # mean_distance /= batch_count
            # mean_loss /= batch_count

            self.logger.info(
                f"Iteration {it+1}/{self.num_iterations}, Mean Distance: {mean_distance}, Mean Loss: {mean_loss}"
            )

            if self.use_lagrangian:
                self.logger.info(f"Lagrangian Multipliers - mu: {self.mu.item():.6f}, nu: {self.nu.item():.6f}")

            det_mask = []
            for lamb in self.lambs:
                lamb.eval()
                det_mask.append(lamb().item())
            self.logger.info(f"Deterministic Mask: {det_mask}")
            for lamb in self.lambs:
                lamb.train()

    def get_binary_mask(self) -> List[int]:
        """
        Converts the learned masks to binary masks and identifies pruned layers.

        Returns:
        --------
        - List of indices indicating pruned layers.
        """
        smooth_mask = [lamb().item() for lamb in self.lambs]
        binary_mask = [1 if mask > 0.5 else 0 for mask in smooth_mask]
        # kept_layers = [idx for idx, mask in enumerate(binary_mask) if mask == 1]
        return binary_mask
