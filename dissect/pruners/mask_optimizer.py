# mask_optimizer.py
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
        skipped_layers: List[str] = None,  # New parameter
        weight_lm_loss: float = 1.0,  # New parameter for LM loss weight
    ):
        """
        Initializes the MaskOptimizer with required parameters.

        Parameters:
        -----------
        - model: The neural network model to prune.
        - target_modules: List of module names to apply masks.
        - outputs: Original model output logits for comparison.
        - data_loader: DataLoader for pruning dataset.
        - distance_metric: Metric to measure output distance. Can be 'norm', 'angular_distance', 'kl_divergence', 'js_divergence', or 'combined'.
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
        - use_lagrangian_proxy: Flag to use the loss for Lagrangian optimization but fixing the multipliers.
        - gradient_checkpointing: Flag to enable gradient checkpointing.
        - warm_up: Number of iterations before using regularization.
        - skipped_layers: List of layer names to skip from optimization and pruning.
        - weight_lm_loss: Weight for the language modeling loss in the combined distance metric.
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
        self.skipped_layers = skipped_layers if skipped_layers is not None else []
        self.weight_lm_loss = weight_lm_loss  # Initialize LM loss weight

        self.lambs = []
        self.target_layers_ordered = []  # To maintain order and track skipped layers
        self._initialize_masks()
        self.initial_lr_mask = lr_mask  # Store initial learning rate
        self.initial_gamma = gamma  # Store initial gamma
        self.initial_eta = eta  # Store initial eta

        params_to_optimize = []
        for lamb in self.lambs:
            lamb.train()
            params_to_optimize.extend(list(lamb.parameters()))
        self.optimizer_mask = torch.optim.Adam(params_to_optimize, lr=self.lr_mask)

        if self.use_lagrangian:
            # Initialize Lagrangian multipliers as Parameters for optimization
            self.mu = nn.Parameter(torch.tensor(self.gamma, device=self.device))
            self.nu = nn.Parameter(torch.tensor(self.eta, device=self.device))
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
        Handles skipped layers by excluding them from mask optimization.
        """
        for name, module in self.model.named_modules():
            for target_module in self.target_modules:
                if target_module in name.split(".")[-1]:
                    if name in self.skipped_layers:
                        # Skipped layers are not added to mask optimization
                        self.target_layers_ordered.append((name, True))
                        continue
                    # Register L0Mask for optimization
                    lamb = L0Mask(
                        shape=(1,), temperature=2.0 / 3.0, droprate_init=self.target_sparsity, device=self.device
                    )
                    module.register_forward_hook(self._create_concrete_mask_hook(lamb))
                    self.lambs.append(lamb)
                    self.target_layers_ordered.append((name, False))

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

    def _compute_distance(self, original_output: torch.Tensor, output_logits: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the distance between original and current model outputs based on the selected metric.
        If the distance metric is 'combined', it also includes the language modeling loss.

        Parameters:
        -----------
        - original_output: Original model output logits.
        - output_logits: Current model output logits.
        - labels: Ground truth labels for language modeling loss (optional).

        Returns:
        --------
        - Computed distance.
        """
        if self.distance_metric == "norm":
            return F.mse_loss(original_output, output_logits)
        elif self.distance_metric == "angular_distance":
            cosine_similarity = F.cosine_similarity(original_output, output_logits, dim=-1)
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
        elif self.distance_metric == "combined":
            if labels is None:
                raise ValueError("Labels must be provided for the 'combined' distance metric.")
            # Compute the original distance (using JS Divergence)
            mean_prob = 0.5 * (F.softmax(original_output, dim=-1) + F.softmax(output_logits, dim=-1))
            distance = 0.5 * (
                F.kl_div(F.log_softmax(original_output, dim=-1), mean_prob, reduction="mean")
                + F.kl_div(F.log_softmax(output_logits, dim=-1), mean_prob, reduction="mean")
            )
            # Compute Language Modeling Loss (Next Token Cross-Entropy)
            lm_loss = F.cross_entropy(output_logits.view(-1, output_logits.size(-1)), labels.view(-1).to(output_logits.device))
            # Combine both losses
            combined_loss = distance + self.weight_lm_loss * lm_loss
            return combined_loss
        else:
            raise NotImplementedError(f"Unsupported distance metric: {self.distance_metric}")

    def optimize_masks(self):
        """
        Runs the optimization loop to learn the masks using a multistage approach.
        """
        self.model.train()
        total_layers = len(self.target_layers_ordered)
        target_total_to_prune = self.target_sparsity * total_layers

        for it in range(self.num_iterations):
            # Adjust learning rate and regularization weights based on iteration
            if it < self.warm_up:
                # Warm-up phase
                current_lr_mask = self.initial_lr_mask * 0.1  # Set lr_mask to 10% of initial value
                current_gamma = 0.0  # No regularization during warm-up
                current_eta = 0.0
            else:
                # Post warm-up phase
                current_lr_mask = self.initial_lr_mask  # Use initial learning rate
                # Increase regularization weights over iterations
                progress = (it - self.warm_up) / (self.num_iterations - self.warm_up)
                current_gamma = self.initial_gamma * (1 + 9 * progress)  # Scale gamma up to 10x
                current_eta = self.initial_eta * (1 + 9 * progress)  # Scale eta up to 10x

            # Update optimizer learning rate
            for param_group in self.optimizer_mask.param_groups:
                param_group['lr'] = current_lr_mask

            self.optimizer_mask.zero_grad()
            if self.use_lagrangian:
                self.optimizer_mu.zero_grad()

            mean_distance = []
            mean_loss = []

            with suppress_output(), suppress_tqdm():
                for (data, original_output) in alive_it(
                    zip(self.data_loader, self.outputs), total=len(self.data_loader), enrich_print=False, disable=True
                ):
                    labels = data.get("labels")
                    original_output = original_output.to(self.device)
                    data = BatchEncoding(data).to(self.device)
                    output = self.model(**data)
                    output_logits = output.logits.to(self.device)

                    distance = self._compute_distance(original_output, output_logits, labels)

                    # Compute L0 norms for all masks
                    l0_tensor = torch.cat([lamb.l0_norm() for lamb in self.lambs])

                    # Constraint: sum(l0_tensor) == target_total
                    constraint = len(self.lambs) - torch.sum(l0_tensor) - target_total_to_prune

                    if self.use_lagrangian:
                        # Lagrangian Loss: L = similarity_loss + mu * constraint + nu * constraint^2
                        loss = distance + self.mu * constraint + self.nu * (constraint**2)
                    elif self.use_lagrangian_proxy:
                        if it < self.warm_up:
                            loss = distance
                        else:
                            # Use current_gamma and current_eta
                            loss = distance + current_gamma * torch.abs(constraint) + current_eta * (constraint**2)
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
                        if self.use_lagrangian:
                            self.accelerator.backward(loss, retain_graph=True)
                        else:
                            self.accelerator.backward(loss)
                    else:
                        if self.use_lagrangian:
                            loss.backward(retain_graph=True)
                        else:
                            loss.backward()

                    # Update mask parameters
                    self.optimizer_mask.step()

                    if self.use_lagrangian and it >= self.warm_up:
                        # Update Lagrangian multipliers via gradient ascent
                        # Compute the Lagrangian components separately
                        # Loss with respect to multipliers: mu * constraint + nu * constraint^2
                        loss_mu = self.mu * constraint + self.nu * (constraint**2)
                        # To maximize loss_mu, minimize -loss_mu
                        if self.gradient_checkpointing:
                            self.accelerator.backward(-loss_mu)
                        else:
                            (-loss_mu).backward()
                        self.optimizer_mu.step()

                        # Detach multipliers to prevent backpropagation through them in mask optimizer
                        self.mu.data.clamp_(min=0.0)  # Ensure mu is non-negative
                        self.nu.data.clamp_(min=0.0)  # Ensure nu is non-negative

                    mean_distance.append(distance.item())
                    mean_loss.append(loss.item())

            mean_distance = sum(mean_distance) / len(mean_distance)
            mean_loss = sum(mean_loss) / len(mean_loss)

            self.logger.info(
                f"Iteration {it+1}/{self.num_iterations}, "
                f"Mean Distance: {mean_distance}, Mean Loss: {mean_loss}, "
                f"LR Mask: {current_lr_mask}, Gamma: {current_gamma}, Eta: {current_eta}"
            )

            if self.use_lagrangian:
                self.logger.info(f"Lagrangian Multipliers - mu: {self.mu.item()}, nu: {self.nu.item()}")

            # Generate deterministic mask including skipped layers
            det_mask = []
            lamb_index = 0  # Initialize a separate index for lambs
            for _, is_skipped in self.target_layers_ordered:
                if is_skipped:
                    det_mask.append(1)
                else:
                    self.lambs[lamb_index].eval()
                    det_mask.append(self.lambs[lamb_index]().item())
                    self.lambs[lamb_index].train()
                    lamb_index += 1

            self.logger.info(f"Deterministic Mask: {det_mask}")

    def get_binary_mask(self) -> List[int]:
        """
        Converts the learned masks to binary masks and identifies pruned layers.

        Returns:
        --------
        - List of indices indicating pruned layers.
        """
        binary_mask = []
        lamb_index = 0  # Initialize a separate index for lambs
        for _, is_skipped in self.target_layers_ordered:
            if is_skipped:
                binary_mask.append(1)
            else:
                self.lambs[lamb_index].eval()
                mask_val = self.lambs[lamb_index]().item()
                binary_mask.append(1 if mask_val > 0.5 else 0)
                lamb_index += 1
        return binary_mask
