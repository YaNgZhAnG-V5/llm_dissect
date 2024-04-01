from typing import Dict, List, Tuple

import torch


class BinaryMaskMixin:

    @staticmethod
    def _get_global_binary_mask(
        concat_stats: torch.Tensor, sparsity: float, split_size: List[int]
    ) -> List[torch.Tensor]:
        """get global binary masks for the model given the sparsity rate"""
        top_k = int(concat_stats.numel() * (1 - sparsity))
        _, top_k_inds = torch.topk(concat_stats, top_k, sorted=False, largest=True)
        binary_mask = torch.zeros_like(concat_stats, dtype=torch.bool)
        binary_mask.scatter_(dim=-1, index=top_k_inds, src=torch.ones_like(top_k_inds, dtype=torch.bool))
        global_binary_masks = binary_mask.split(dim=-1, split_size=split_size)
        return global_binary_masks

    @staticmethod
    def _get_local_binary_masks(flatten_stats: List[torch.Tensor], sparsity: float) -> List[torch.Tensor]:
        """get local binary masks for each layer given the sparsity rate"""
        local_binary_masks = []
        for stat in flatten_stats:
            top_k = int(stat.numel() * (1 - sparsity))
            _, top_k_inds = torch.topk(stat, top_k, sorted=False, largest=True)
            local_binary_mask = torch.zeros_like(stat, dtype=torch.bool)
            local_binary_mask.scatter_(dim=-1, index=top_k_inds, src=torch.ones_like(top_k_inds, dtype=torch.bool))
            local_binary_masks.append(local_binary_mask)
        return local_binary_masks

    def global_prune(
        self,
        sparsity: float,
        flatten_stats: List[torch.Tensor],
        concat_stats: torch.Tensor,
        split_size: List[int],
        shape_info: Dict[str, Tuple[torch.Size, int]],
    ) -> Dict[str, torch.Tensor]:
        """prune the model globally (rank all neurons diregard their belonging layers) with the sparsity rate"""
        mask_state_dict = dict()
        global_binary_masks = self._get_global_binary_mask(
            concat_stats=concat_stats, sparsity=sparsity, split_size=split_size
        )
        for i, (layer_name, (forward_grad_shape, _)) in enumerate(shape_info.items()):
            mask_state_dict.update({layer_name: global_binary_masks[i].reshape(forward_grad_shape)})
        return mask_state_dict

    def local_prune(
        self,
        sparsity: float,
        flatten_stats: List[torch.Tensor],
        concat_stats: torch.Tensor,
        split_size: List[int],
        shape_info: Dict[str, Tuple[torch.Size, int]],
    ) -> Dict[str, torch.Tensor]:
        """prune the model locally (rank neurons within each layer) with the sparsity rate"""
        mask_state_dict = dict()
        local_binary_masks = self._get_local_binary_masks(flatten_stats=flatten_stats, sparsity=sparsity)
        for i, (layer_name, (forward_grad_shape, _)) in enumerate(shape_info.items()):
            mask_state_dict.update({layer_name: local_binary_masks[i].reshape(forward_grad_shape)})
        return mask_state_dict

    def global_thres_prune(
        self,
        sparsity: float,
        flatten_stats: List[torch.Tensor],
        concat_stats: torch.Tensor,
        split_size: List[int],
        shape_info: Dict[str, Tuple[torch.Size, int]],
    ) -> Dict[str, torch.Tensor]:
        """global prune with pruning rate thresholding at each layer"""
        mask_state_dict = dict()

        # gat global and local binary masks
        global_binary_masks = self._get_global_binary_mask(
            concat_stats=concat_stats, sparsity=sparsity, split_size=split_size
        )
        local_binary_masks = self._get_local_binary_masks(flatten_stats=flatten_stats, sparsity=sparsity)

        final_binary_masks = []
        for global_binary_mask, local_binary_mask in zip(global_binary_masks, local_binary_masks):
            actual_sparsity = 1 - global_binary_mask.float().sum() / global_binary_mask.numel()
            # enforce per-layer actual sparsity is no greater than the specified sparsity
            if actual_sparsity > sparsity:
                final_binary_mask = local_binary_mask
            else:
                final_binary_mask = global_binary_mask

            final_binary_masks.append(final_binary_mask)

        for i, (layer_name, (forward_grad_shape, _)) in enumerate(shape_info.items()):
            mask_state_dict.update({layer_name: final_binary_masks[i].reshape(forward_grad_shape)})
        return mask_state_dict

    def global_thres_per_head_prune(
        self,
        sparsity: float,
        flatten_stats: List[torch.Tensor],
        concat_stats: torch.Tensor,
        split_size: List[int],
        shape_info: Dict[str, Tuple[torch.Size, int]],
    ) -> Dict[str, torch.Tensor]:
        """global prune with pruning rate thresholding at each layer"""
        mask_state_dict = dict()

        # gat global and local binary masks
        global_binary_masks = self._get_global_binary_mask(
            concat_stats=concat_stats, sparsity=sparsity, split_size=split_size
        )
        local_binary_masks = self._get_local_binary_masks(flatten_stats=flatten_stats, sparsity=sparsity)

        final_binary_masks = []
        for global_binary_mask, local_binary_mask, flatten_stat in zip(
            global_binary_masks, local_binary_masks, flatten_stats
        ):
            actual_sparsity = 1 - global_binary_mask.float().sum() / global_binary_mask.numel()
            # enforce per-layer actual sparsity is no greater than the specified sparsity
            if actual_sparsity > sparsity:
                final_binary_mask = local_binary_mask
            else:
                final_binary_mask = global_binary_mask

            # use the ratio from the global thres for per head pruning
            final_prune_ratio = 1 - final_binary_mask.float().sum() / final_binary_mask.numel()
            final_binary_mask = self.per_head_prune(flatten_stat, final_prune_ratio, num_heads=32)  # TODO: fix head
            final_binary_masks.append(final_binary_mask)

        for i, (layer_name, (forward_grad_shape, _)) in enumerate(shape_info.items()):
            mask_state_dict.update({layer_name: final_binary_masks[i].reshape(forward_grad_shape)})
        return mask_state_dict

    @staticmethod
    def per_head_prune(stats: torch.Tensor, ratio: float, num_heads: int):
        stats = stats.view(num_heads, -1)
        top_k = int(stats.size(1) * (1 - ratio))

        # account for odd number of neurons (due to rotary embedding)
        top_k = top_k + 1 if top_k % 2 == 1 else top_k
        _, top_k_inds = torch.topk(stats, top_k, dim=1, sorted=False, largest=True)
        binary_mask = torch.zeros_like(stats, dtype=torch.bool)
        binary_mask.scatter_(dim=-1, index=top_k_inds, src=torch.ones_like(top_k_inds, dtype=torch.bool))
        return binary_mask.view(-1)
