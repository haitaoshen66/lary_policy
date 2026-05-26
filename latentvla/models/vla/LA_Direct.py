import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from latentvla.models.action_heads import L1RegressionActionHead, ProprioProjector
from latentvla.models.GR00T_ActionHeader import build_flow_gr00t_action_head
from latentvla.models.constants import (
    ACTION_DIM,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM
)
from latentvla.models.vla.utils import _gather_action_token_embeddings, AttentionPooling, gather_non_placeholder_hidden_states

class LA_Direct_VLA(nn.Module):
    def __init__(
        self, vlm, num_images, use_proprio, action_token_id, use_pro_version=False, action_head_type="l1", flow_dit_size="dit-b", prompt_suffix_token_ids=None):
        super().__init__()
        self.vlm = vlm
        self.action_token_id = action_token_id
        self.num_images = num_images
        self.action_head_type = action_head_type
        self.prompt_suffix_token_ids = prompt_suffix_token_ids
        self._flow_debug_printed = False
        
        for param in self.vlm.parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        self.use_proprio = use_proprio
        self.vlm = get_peft_model(self.vlm, lora_config)

        if self.action_head_type == "l1":
            self.proprio_projector = ProprioProjector(
                llm_dim = 2048,
                proprio_dim = PROPRIO_DIM
            )
            self.action_head = L1RegressionActionHead(
                input_dim = 2048,
                hidden_dim = 2048,
                action_dim = ACTION_DIM,
                num_blocks = 11,
                num_action_chunk = NUM_ACTIONS_CHUNK,
                use_pro_version = use_pro_version
            )
        elif self.action_head_type == "flow_gr00t":
            self.proprio_projector = None
            self.action_head = build_flow_gr00t_action_head(
                vlm_hidden_size=2048,
                action_dim=ACTION_DIM,
                action_horizon=NUM_ACTIONS_CHUNK,
                state_dim=PROPRIO_DIM if self.use_proprio else 0,
                flow_dit_size=flow_dit_size,
            )
        else:
            raise ValueError(f"Unsupported action_head_type: {self.action_head_type}")
        self.pooling = AttentionPooling(hidden_dim=2048)
        

    def forward(self, batch, training=True):
        B, N, _ = batch["image_grid_thw"].shape
        image_grid_thw = batch["image_grid_thw"].reshape(B*N, 3)
        vlm_outputs = self.vlm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            labels=batch["labels"],
        )
        vlm_loss = vlm_outputs.loss
        #print(vlm_loss)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ground_truth_actions = batch["actions"].to(torch.bfloat16)
            if self.action_head_type == "flow_gr00t":
                gather_ret = gather_non_placeholder_hidden_states(
                    last_hidden=vlm_outputs.hidden_states[-1],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    placeholder_token_id=self.action_token_id,
                    prompt_suffix_token_ids=self.prompt_suffix_token_ids,
                    return_debug_stats=not self._flow_debug_printed,
                )
                if self._flow_debug_printed:
                    vl_embs, vl_mask = gather_ret
                else:
                    vl_embs, vl_mask, debug_stats = gather_ret
                    print(
                        "[flow_gr00t debug][la_direct] "
                        f"valid={debug_stats['valid_counts'].tolist()} "
                        f"kept={debug_stats['kept_counts'].tolist()} "
                        f"placeholder={debug_stats['placeholder_counts'].tolist()} "
                        f"suffix_removed={debug_stats['suffix_removed_counts'].tolist()}",
                        flush=True,
                    )
                    self._flow_debug_printed = True
                state = batch["proprio"].unsqueeze(1).to(torch.bfloat16) if self.use_proprio else None
                if training:
                    action_loss = self.action_head(
                        vl_embs=vl_embs,
                        actions=ground_truth_actions,
                        state=state,
                        encoder_attention_mask=vl_mask,
                    )
                else:
                    predicted_actions = self.action_head.predict_action(
                        vl_embs=vl_embs,
                        state=state,
                        encoder_attention_mask=vl_mask,
                    )
                    action_loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)
            else:
                num_patches = 256 * self.num_images
                multi_layer_hidden_states = []
                
                for layer_hidden in vlm_outputs.hidden_states[-12:]:
                    B, L, H = layer_hidden.shape
                    image_hidden = layer_hidden[:, :num_patches]
                    text_hidden = layer_hidden

                    action_hidden = _gather_action_token_embeddings(
                        last_hidden=text_hidden,
                        input_ids=batch["input_ids"][:, :],
                        action_token_id=self.action_token_id,
                        num_chunk=NUM_ACTIONS_CHUNK*4,
                    )
                    action_hidden = action_hidden.reshape(B, NUM_ACTIONS_CHUNK, 4, H)
                    action_hidden = self.pooling(action_hidden)

                    image_latent = image_hidden.unsqueeze(1)
                    action_latent = action_hidden.unsqueeze(1)

                    all_hidden = torch.cat((image_latent, action_latent), dim=2)
                    multi_layer_hidden_states.append(all_hidden)
                
                multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim = 1)
                predicted_actions = self.action_head.predict_action(
                    multi_layer_hidden_states,
                    proprio=batch["proprio"] if self.use_proprio else None,
                    proprio_projector=self.proprio_projector if self.use_proprio else None,
                    phase="Training" if training else "Inference",
                )
                action_loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)
        return dict(
            action_loss=action_loss,
            action_token_loss=vlm_loss
        )
