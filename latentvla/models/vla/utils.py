import torch
import torch.nn as nn
from latentvla.models.constants import NUM_ACTIONS_CHUNK

def _gather_action_token_embeddings(
        last_hidden: torch.Tensor,   # [B, L, H]
        input_ids: torch.Tensor,     # [B, L]
        action_token_id=None,
        num_chunk=NUM_ACTIONS_CHUNK,
    ) -> torch.Tensor:
        if action_token_id is None:
            raise ValueError("action_token_id should be provided.")

        device = input_ids.device
        B, L, H = last_hidden.shape

        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            mask = torch.isin(input_ids, id_list)
        else:
            mask = (input_ids == action_token_id)  # [B, L]

        counts = mask.sum(dim=1)  # [B]
        if (counts < num_chunk).any():
            insufficient = (counts < num_chunk).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"tokens not enough {num_chunk}: {insufficient} | counts={counts.tolist()}"
            )

        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))

        topk_pos = masked_pos.topk(k=num_chunk, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values                     # [B, chunk_len]

        # Gather
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)   # [B, chunk_len, H]
        action_queries = last_hidden.gather(dim=1, index=expanded_index)  # [B, chunk_len, H]
        return action_queries

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, h):  
        scores = torch.matmul(h, self.query)
        weights = torch.softmax(scores, dim=1)  # [B, 4]
        pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)

        return pooled


def gather_non_placeholder_hidden_states(
    last_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    placeholder_token_id: int,
    prompt_suffix_token_ids=None,
    return_debug_stats: bool = False,
):
    """
    Keep only valid instruction/image tokens from the last hidden states by
    removing placeholders and, when provided, the fixed action-prompt suffix
    that appears immediately before the placeholder run.

    Returns:
        packed_hidden: [B, L_keep_max, H]
        packed_mask:   [B, L_keep_max] with 1 for kept tokens
    """
    keep_mask = attention_mask.bool() & input_ids.ne(placeholder_token_id)

    if prompt_suffix_token_ids is not None:
        suffix_ids = torch.as_tensor(
            prompt_suffix_token_ids,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        suffix_len = int(suffix_ids.numel())
        if suffix_len > 0:
            for batch_idx in range(input_ids.shape[0]):
                placeholder_pos = torch.nonzero(
                    attention_mask[batch_idx].bool() & input_ids[batch_idx].eq(placeholder_token_id),
                    as_tuple=False,
                ).flatten()
                if placeholder_pos.numel() == 0:
                    continue

                first_placeholder = int(placeholder_pos[0].item())
                start = first_placeholder - suffix_len
                if start < 0:
                    continue

                candidate = input_ids[batch_idx, start:first_placeholder]
                if torch.equal(candidate, suffix_ids):
                    keep_mask[batch_idx, start:first_placeholder] = False

    valid_counts = attention_mask.bool().sum(dim=1)
    keep_counts = keep_mask.sum(dim=1)
    if (keep_counts == 0).any():
        bad = (keep_counts == 0).nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(f"All tokens were filtered out for samples: {bad}")

    bsz, _, hidden_dim = last_hidden.shape
    max_kept = int(keep_counts.max().item())
    packed_hidden = last_hidden.new_zeros((bsz, max_kept, hidden_dim))
    packed_mask = attention_mask.new_zeros((bsz, max_kept))

    placeholder_counts = (attention_mask.bool() & input_ids.eq(placeholder_token_id)).sum(dim=1)
    suffix_removed_counts = valid_counts - keep_counts - placeholder_counts

    for batch_idx in range(bsz):
        selected_hidden = last_hidden[batch_idx][keep_mask[batch_idx]]
        kept = selected_hidden.shape[0]
        packed_hidden[batch_idx, :kept] = selected_hidden
        packed_mask[batch_idx, :kept] = 1

    if return_debug_stats:
        debug_stats = {
            "valid_counts": valid_counts.detach().cpu(),
            "kept_counts": keep_counts.detach().cpu(),
            "placeholder_counts": placeholder_counts.detach().cpu(),
            "suffix_removed_counts": suffix_removed_counts.detach().cpu(),
        }
        return packed_hidden, packed_mask, debug_stats

    return packed_hidden, packed_mask
