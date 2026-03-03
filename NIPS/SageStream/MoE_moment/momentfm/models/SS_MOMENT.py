import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.SA_MoE import SA_MoEFactory, SA_MoE
from typing import Optional, Dict, Any, Tuple

# Import IIB modules - try from sagestream_refactored first, then relative
try:
    from sagestream_refactored.models.iib import IIB
    from sagestream_refactored.models.iib_icml import IIB_ICML
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../sagestream_refactored'))
    from models.iib import IIB
    from models.iib_icml import IIB_ICML

class FeatureCollector:
    def __init__(self):
        self.gamma_params = []
        self.beta_params = []
        self.pre_alignment_features = []
        self.post_alignment_features = []
        self.layer_ids = []

    def clear(self):
        self.gamma_params.clear()
        self.beta_params.clear()
        self.pre_alignment_features.clear()
        self.post_alignment_features.clear()
        self.layer_ids.clear()

    def collect_layer_data(self,layer_id,gamma,beta,pre_features,post_features):
        self.layer_ids.append(layer_id)
        self.gamma_params.append(gamma.clone().detach())
        self.beta_params.append(beta.clone().detach())
        self.pre_alignment_features.append(pre_features.clone().detach())
        self.post_alignment_features.append(post_features.clone().detach())


class EnhancedClassificationOutput:
    def __init__(self,logits,aux_loss,gamma_params=None,beta_params=None,
                 pre_alignment_features=None,post_alignment_features=None,layer_ids=None):
        self.logits = logits
        self.aux_loss = aux_loss
        self.gamma_params = gamma_params or []
        self.beta_params = beta_params or []
        self.pre_alignment_features = pre_alignment_features or []
        self.post_alignment_features = post_alignment_features or []
        self.layer_ids = layer_ids or []


class OptimizedGlobalExplicitDecouplingT5Block(nn.Module):
    def __init__(self,config,layer_id,has_relative_attention_bias=False,
                 optimized_shared_factory=None,shared_config=None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        
        from transformers.models.t5.modeling_t5 import T5LayerSelfAttention, T5LayerFF
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        
        self.layer.append(T5LayerFF(config))
        
        if optimized_shared_factory is not None:
            top_k = shared_config.get('top_k', 2) if shared_config else 2
            aux_loss_weight = shared_config.get('aux_loss_weight', 0.01) if shared_config else 0.01

            enable_hypernetwork = (
                shared_config.get('enable_shared_backbone_hypernetwork', False) or
                shared_config.get('enable_subject_style_normalization', False)
            ) if shared_config else False

            num_subjects = shared_config.get('num_subjects', 50) if shared_config else 50
            subject_embedding_dim = shared_config.get('subject_embedding_dim', 64) if shared_config else 64
            expert_embedding_dim = shared_config.get('expert_embedding_dim', 32) if shared_config else 32
            hyper_expert_hidden_dim = shared_config.get('hyper_expert_hidden_dim', 64) if shared_config else 64
            num_channels = shared_config.get('num_channels', 16) if shared_config else 16

            self.shared_knowledge = optimized_shared_factory.create_module(
                layer_id=layer_id,
                top_k=top_k,
                aux_loss_weight=aux_loss_weight,
                enable_shared_backbone_hypernetwork=enable_hypernetwork,
                num_subjects=num_subjects,
                subject_embedding_dim=subject_embedding_dim,
                expert_embedding_dim=expert_embedding_dim,
                hyper_expert_hidden_dim=hyper_expert_hidden_dim,
                num_channels=num_channels
            )
        else:
            self.shared_knowledge = None
            
        self.adaptive_knowledge = None
            
        self.gating_network = None
            
    def set_training_stage(self,stage):
        if stage == "source_domain":
            if self.shared_knowledge is not None:
                self.shared_knowledge.unfreeze_parameters()
            if self.adaptive_knowledge is not None:
                self.adaptive_knowledge.set_active(False)
                self.adaptive_knowledge.freeze_parameters()
                
        elif stage == "tta":
            if self.shared_knowledge is not None:
                self.shared_knowledge.freeze_parameters()
            if self.adaptive_knowledge is not None:
                self.adaptive_knowledge.set_active(True)
                self.adaptive_knowledge.unfreeze_parameters()
        else:
            raise ValueError(f"Unknown training stage: {stage}")
    
    def forward(self, hidden_states, attention_mask=None, position_bias=None,
                encoder_hidden_states=None, encoder_attention_mask=None, encoder_decoder_position_bias=None,
                layer_head_mask=None, cross_attn_layer_head_mask=None,
                past_key_value=None, past_key_values=None, use_cache=False,
                output_attentions=False, return_dict=True, cache_position=None, *args, **kwargs):
        # Build self-attention kwargs compatible with both old and new transformers
        # Ensure falsy values (False, 0, etc.) are normalized to None
        _past = past_key_values or past_key_value or None
        import inspect
        _sa_params = inspect.signature(self.layer[0].forward).parameters
        sa_kwargs = dict(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        if 'past_key_values' in _sa_params:       # transformers >= 5.x
            sa_kwargs['past_key_values'] = _past
        elif 'past_key_value' in _sa_params:       # transformers <= 4.x
            sa_kwargs['past_key_value'] = _past
        if 'layer_head_mask' in _sa_params:
            sa_kwargs['layer_head_mask'] = layer_head_mask
        if 'cache_position' in _sa_params:
            sa_kwargs['cache_position'] = cache_position
        self_attn_outputs = self.layer[0](**sa_kwargs)
        attn_output = self_attn_outputs[0]
        outputs = self_attn_outputs[1:]

        if self.shared_knowledge is not None:
            ffn_output = self.layer[1](attn_output)

            current_subject_ids = getattr(self.shared_knowledge, '_current_subject_ids', None)

            shared_transform = self.shared_knowledge(attn_output, subject_ids=current_subject_ids)

            adaptive_transform = None
            if self.adaptive_knowledge is not None and self.adaptive_knowledge.is_active:
                adaptive_transform = self.adaptive_knowledge(attn_output)

            if adaptive_transform is not None:
                final_output = ffn_output + shared_transform + adaptive_transform
            else:
                final_output = shared_transform
        else:
            final_output = self.layer[1](attn_output)

        outputs = (final_output,) + outputs

        return outputs
    
    def get_aux_loss(self):
        aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        if self.shared_knowledge is not None:
            aux_loss += self.shared_knowledge.get_aux_loss()
            
        if self.adaptive_knowledge is not None:
            aux_loss += self.adaptive_knowledge.get_aux_loss()
            
        return aux_loss
    
    def clear_aux_losses(self):
        if self.shared_knowledge is not None:
            self.shared_knowledge.clear_aux_losses()
            
        if self.adaptive_knowledge is not None:
            self.adaptive_knowledge.clear_aux_losses()
    
    def get_knowledge_analysis(self,x):
        analysis = {
            'layer_id': self.layer_id,
            'shared_knowledge': {},
            'adaptive_knowledge': {},
            'optimized_pool_efficiency': {}
        }
        
        if self.shared_knowledge is not None:
            analysis['shared_knowledge'] = self.shared_knowledge.get_expert_stats()
            analysis['optimized_pool_efficiency'] = self.shared_knowledge.get_parameter_efficiency_metrics()
            
        if self.adaptive_knowledge is not None:
            analysis['adaptive_knowledge'] = self.adaptive_knowledge.get_expert_stats()
        
        return analysis


class SageStream(nn.Module):
    def __init__(self,config,decoupling_config=None):
        super().__init__()
        self.config = config
        self.decoupling_config = decoupling_config or {}

        from .moment import MOMENT
        base_moment = MOMENT(config)

        self.patch_embedding = base_moment.patch_embedding
        self.normalizer = base_moment.normalizer
        self.tokenizer = base_moment.tokenizer
        self.mask_generator = base_moment.mask_generator
        self.seq_len = config.seq_len
        self.patch_len = config.patch_len

        self._patch_normalizer()

        self.encoder = base_moment.encoder
        encoder_config = self.encoder.config

        # Check if SA-MoE is enabled
        self.use_moe = self.decoupling_config.get('use_moe', True)

        shared_config = self.decoupling_config.get('shared_config', {})

        if self.use_moe:
            self.optimized_shared_factory = SA_MoEFactory(
                d_model=encoder_config.d_model,
                d_ff=encoder_config.d_ff,
                num_experts=shared_config.get('num_experts', 4),
                dropout=shared_config.get('dropout', 0.1),
                freq_learning_mode=shared_config.get('freq_learning_mode', 'adaptive_filter'),
                routing_strategy=shared_config.get('routing_strategy', 'frequency_aware'),
                expert_dim_ratio=shared_config.get('expert_dim_ratio', 1.0),
                max_freq=shared_config.get('max_freq', 40.0),
                sampling_rate=shared_config.get('sampling_rate', 256.0)
            )

            for i, original_block in enumerate(self.encoder.block):
                has_relative_attention_bias = (i == 0)

                decoupling_block = OptimizedGlobalExplicitDecouplingT5Block(
                    config=encoder_config,
                    layer_id=i,
                    has_relative_attention_bias=has_relative_attention_bias,
                    optimized_shared_factory=self.optimized_shared_factory,
                    shared_config=shared_config
                )

                decoupling_block.layer[0].load_state_dict(original_block.layer[0].state_dict())
                decoupling_block.layer[1].load_state_dict(original_block.layer[1].state_dict())

                self.encoder.block[i] = decoupling_block
        else:
            # No SA-MoE: keep original encoder blocks, use a dummy factory
            self.optimized_shared_factory = None

        self.head = None

        self.current_stage = "source_domain"

        # ==========================================
        # IIB (Information Invariant Bottleneck) Module
        # ==========================================
        iib_config = self.decoupling_config.get('iib_config', {})
        self.enable_iib = iib_config.get('enable_iib', False)
        self.iib_variant = iib_config.get('iib_variant', 'nips')

        if self.enable_iib:
            if self.iib_variant == 'icml':
                # ICML variant: dual-head + CI loss
                self.iib = IIB_ICML(
                    input_dim=encoder_config.d_model,
                    latent_dim=iib_config.get('latent_dim', 256),
                    hidden_dim=iib_config.get('icml_head_hidden_dim', 16),
                    num_classes=iib_config.get('num_classes', 2),
                    domain_dim=iib_config.get('icml_domain_dim', 1),
                    lambda_ib=iib_config.get('icml_lambda_ib', 0.1),
                    beta_ci=iib_config.get('icml_beta_ci', 10.0),
                )
                # ICML does not use separate loss weights at SageStream level;
                # total_iib_loss is computed inside IIB_ICML.compute_losses()
                self.iib_kl_loss_weight = 0.0
                self.iib_adv_loss_weight = 0.0
                self.use_grl_schedule = False
                self.current_epoch = 0
            else:
                # NIPS variant: GRL + Subject Discriminator
                self.iib = IIB(
                    input_dim=encoder_config.d_model,
                    hidden_dim=iib_config.get('hidden_dim', 256),
                    latent_dim=iib_config.get('latent_dim', 256),
                    num_subjects=iib_config.get('num_subjects', 50),
                    grl_alpha=iib_config.get('grl_alpha', 1.0),
                    discriminator_hidden_dim=iib_config.get('discriminator_hidden_dim', None),
                    dropout=iib_config.get('dropout', 0.1)
                )
                # Store IIB config for loss weight access
                self.iib_kl_loss_weight = iib_config.get('kl_loss_weight', 0.1)
                self.iib_adv_loss_weight = iib_config.get('adv_loss_weight', 0.1)

                # GRL schedule parameters
                self.use_grl_schedule = iib_config.get('use_grl_schedule', False)
                self.grl_alpha_max = iib_config.get('grl_alpha_max', 1.0)
                self.grl_schedule_epochs = iib_config.get('grl_schedule_epochs', 10)
                self.current_epoch = 0
        else:
            self.iib = None
            self.iib_variant = 'none'
            self.iib_kl_loss_weight = 0.0
            self.iib_adv_loss_weight = 0.0
    
    def _patch_normalizer(self):
        original_get_statistics = self.normalizer._get_statistics
        
        def patched_get_statistics(x, mask=None):
            if mask is None:
                mask = torch.ones((x.shape[0], x.shape[-1]), device=x.device)
            n_channels = x.shape[1]
            mask = mask.unsqueeze(1).repeat(1, n_channels, 1).bool()
            masked_x = torch.where(mask, x, torch.nan)
            self.normalizer.mean = torch.nanmean(masked_x, dim=-1, keepdim=True).detach()
            
            tensor_mean = masked_x.nanmean(dim=-1, keepdim=True)
            output = (masked_x - tensor_mean).square().nanmean(dim=-1, keepdim=True)
            self.normalizer.stdev = output.sqrt().detach() + self.normalizer.eps
        
        self.normalizer._get_statistics = patched_get_statistics
        
    def set_training_stage(self,stage):
        self.current_stage = stage

        if self.use_moe:
            for block in self.encoder.block:
                if isinstance(block, OptimizedGlobalExplicitDecouplingT5Block):
                    block.set_training_stage(stage)

            if stage == "source_domain":
                self.optimized_shared_factory.unfreeze_all_experts()
                self.optimized_shared_factory.unfreeze_shared_router()
                self.optimized_shared_factory.unfreeze_all_modules()
            elif stage == "tta":
                self.optimized_shared_factory.freeze_all_experts()
                self.optimized_shared_factory.freeze_shared_router()
                self.optimized_shared_factory.freeze_all_modules()

        if stage == "source_domain":
            # Unfreeze IIB if enabled
            if self.iib is not None:
                for param in self.iib.parameters():
                    param.requires_grad = True
        elif stage == "tta":
            # Freeze IIB during TTA
            if self.iib is not None:
                for param in self.iib.parameters():
                    param.requires_grad = False

    def set_epoch(self, epoch: int):
        """Set current epoch for GRL schedule."""
        self.current_epoch = epoch
        if self.enable_iib and self.use_grl_schedule and self.iib is not None:
            # Progressive GRL alpha schedule
            progress = min(1.0, epoch / self.grl_schedule_epochs)
            current_alpha = self.grl_alpha_max * progress
            self.iib.set_grl_alpha(current_alpha)
    
    def forward(self,x_enc,mask=None,subject_ids=None):
        if mask is not None:
            mask = mask.to(x_enc.device)
        x_enc = self.normalizer(x_enc, mode="norm", mask=mask)

        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x_enc)

        if mask is None:
            actual_seq_len = x_enc.shape[2] * self.patch_len
            mask = torch.ones((x_enc.shape[0], actual_seq_len), device=x_enc.device)

        x_enc = self.patch_embedding(x_enc, mask=mask)

        batch_size, n_channels, n_patches, d_model = x_enc.shape
        x_enc = x_enc.reshape(batch_size * n_channels, n_patches, d_model)

        if mask is not None:
            from momentfm.utils.masking import Masking
            patch_view_mask = Masking.convert_seq_to_patch_view(mask, self.patch_len)
            attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        else:
            attention_mask = None

        if subject_ids is not None and self.optimized_shared_factory is not None:
            self.optimized_shared_factory.set_subject_ids_for_all_modules(subject_ids)

        encoder_outputs = self.encoder(inputs_embeds=x_enc, attention_mask=attention_mask, use_cache=False)

        last_hidden_state = encoder_outputs.last_hidden_state

        # ==========================================
        # Apply IIB (Information Invariant Bottleneck)
        # ==========================================
        if self.enable_iib and self.iib is not None:
            # Apply IIB to encoder outputs before reshaping
            # last_hidden_state: (batch_size * n_channels, n_patches, d_model)
            iib_output = self.iib(last_hidden_state)
            last_hidden_state = iib_output

        last_hidden_state = last_hidden_state.reshape(batch_size, n_channels, n_patches, d_model)

        return last_hidden_state.mean(dim=1)
    
    def get_total_aux_loss(self):
        total_aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        if not self.use_moe:
            return total_aux_loss

        for block in self.encoder.block:
            if isinstance(block, OptimizedGlobalExplicitDecouplingT5Block):
                aux_loss = block.get_aux_loss()
                if aux_loss > 0:
                    total_aux_loss = total_aux_loss + aux_loss

        return total_aux_loss

    def get_iib_losses(self, subject_ids: Optional[torch.Tensor] = None,
                       labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute IIB losses.

        For NIPS variant: returns kl_loss, adv_loss
        For ICML variant: returns inv_loss, env_loss, ib_loss, ci_loss, total_iib_loss

        Args:
            subject_ids: Ground truth subject IDs
            labels: Ground truth class labels (required for ICML variant)

        Returns:
            Dictionary of loss components
        """
        if self.enable_iib and self.iib is not None:
            if self.iib_variant == 'icml':
                if labels is not None and subject_ids is not None:
                    return self.iib.compute_losses(labels, subject_ids)
                else:
                    device = next(self.parameters()).device
                    zero = torch.tensor(0.0, device=device)
                    return {
                        'inv_loss': zero, 'env_loss': zero,
                        'ib_loss': zero, 'ci_loss': zero, 'total_iib_loss': zero,
                    }
            else:
                return self.iib.compute_losses(subject_ids)
        else:
            device = next(self.parameters()).device
            return {
                'kl_loss': torch.tensor(0.0, device=device),
                'adv_loss': torch.tensor(0.0, device=device)
            }

    def get_total_iib_loss(self, subject_ids: Optional[torch.Tensor] = None,
                           labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute total IIB loss.

        For NIPS: alpha * KL_loss + beta * adv_loss
        For ICML: total_iib_loss from IIB_ICML (already weighted internally)

        Args:
            subject_ids: Ground truth subject IDs
            labels: Ground truth class labels (required for ICML variant)

        Returns:
            Total IIB loss scalar
        """
        iib_losses = self.get_iib_losses(subject_ids, labels)
        if self.iib_variant == 'icml':
            return iib_losses.get('total_iib_loss', torch.tensor(0.0, device=next(self.parameters()).device))
        else:
            total_iib_loss = (
                self.iib_kl_loss_weight * iib_losses['kl_loss'] +
                self.iib_adv_loss_weight * iib_losses['adv_loss']
            )
            return total_iib_loss

    def clear_all_aux_losses(self):
        if self.use_moe:
            for block in self.encoder.block:
                if isinstance(block, OptimizedGlobalExplicitDecouplingT5Block):
                    block.clear_aux_losses()
        # Clear IIB cache
        if self.iib is not None:
            self.iib.clear_cache()
    
    def get_comprehensive_analysis(self,x_enc):
        analysis = {
            'model_type': 'SageStream',
            'current_stage': self.current_stage,
            'layer_analysis': {},
            'optimized_global_efficiency': {},
            'total_parameters': {},
            'expert_utilization': {},
            'memory_savings': {}
        }
        
        with torch.no_grad():
            _ = self.forward(x_enc)
        
        for i, block in enumerate(self.encoder.block):
            if isinstance(block, OptimizedGlobalExplicitDecouplingT5Block):
                layer_analysis = block.get_knowledge_analysis(x_enc)
                analysis['layer_analysis'][f'layer_{i}'] = layer_analysis
        
        if self.optimized_shared_factory is not None:
            analysis['optimized_global_efficiency'] = self.optimized_shared_factory.get_total_parameter_efficiency()
            analysis['expert_utilization'] = self.optimized_shared_factory.get_optimized_global_pool().get_usage_statistics()
        else:
            analysis['optimized_global_efficiency'] = {}
            analysis['expert_utilization'] = {}
        
        efficiency_report = analysis['optimized_global_efficiency']
        if efficiency_report:
            analysis['memory_savings'] = {
                'parameter_reduction_ratio': efficiency_report.get('overall_parameter_reduction', 0),
                'parameter_savings': efficiency_report.get('overall_parameter_savings', 0),
                'memory_efficiency_score': efficiency_report.get('memory_efficiency_score', 0),
                'router_sharing_enabled': efficiency_report.get('router_sharing_enabled', False),
                'layer_adapters_disabled': not efficiency_report.get('layer_adapters_enabled', False)
            }
        else:
            analysis['memory_savings'] = {}
        
        return analysis
    
    def get_parameter_efficiency_summary(self):
        if self.optimized_shared_factory is not None:
            return self.optimized_shared_factory.get_total_parameter_efficiency()
        return {}
    
    def freeze_pretrained_components(self):
        for param in self.patch_embedding.parameters():
            param.requires_grad = False
        for param in self.normalizer.parameters():
            param.requires_grad = False
        for param in self.tokenizer.parameters():
            param.requires_grad = False

        for block in self.encoder.block:
            if isinstance(block, OptimizedGlobalExplicitDecouplingT5Block):
                for param in block.layer[0].parameters():
                    param.requires_grad = False
                for param in block.layer[1].parameters():
                    param.requires_grad = False
            else:
                # Original T5 blocks (when SA-MoE is disabled)
                for param in block.parameters():
                    param.requires_grad = False
    
    def get_frequency_specialization_analysis(self,x_enc):
        if self.optimized_shared_factory is not None:
            return self.optimized_shared_factory.get_optimized_global_pool().analyze_frequency_specialization(x_enc)
        return {}


class SageStreamPipeline:
    def __init__(self,model,task_name="classification",num_class=2,freeze_pretrained=True,freeze_head=True,pretrained_head_info=None,reduction="concat"):
        self.model = model
        self.task_name = task_name
        self.num_class = num_class
        self.freeze_head = freeze_head
        self.pretrained_head_info = pretrained_head_info
        self.reduction = reduction

        if freeze_pretrained:
            self.model.freeze_pretrained_components()

        self._init_task_head()

    def _create_classification_head(self,input_seq_len=None):
        from torch import nn

        class StandardClassificationHead(nn.Module):
            def __init__(self,n_channels,d_model,n_classes,head_dropout=0.1,reduction="concat"):
                super().__init__()
                self.dropout = nn.Dropout(head_dropout)
                if reduction == "mean":
                    self.linear = nn.Linear(d_model, n_classes)
                elif reduction == "concat":
                    self.linear = nn.Linear(n_channels * d_model, n_classes)
                else:
                    raise ValueError(f"Reduction method {reduction} not implemented. Only 'mean' and 'concat' are supported.")

            def forward(self,x,input_mask=None):
                x = torch.mean(x, dim=1)
                x = self.dropout(x)
                y = self.linear(x)
                return y

        n_channels = getattr(self.model.config, 'n_channels', 16)
        d_model = self.model.config.d_model

        if self.reduction == "concat":
            pass
        elif self.reduction == "mean":
            pass
        else:
            raise ValueError(f"Unsupported reduction method: {self.reduction}. Choose from 'concat' or 'mean'.")

        return StandardClassificationHead(
            n_channels=n_channels,
            d_model=d_model,
            n_classes=self.num_class,
            head_dropout=0.1,
            reduction=self.reduction
        )

    def _init_task_head(self):
        if self.task_name == "classification":
            if self.pretrained_head_info is not None:
                pretrained_num_classes = self.pretrained_head_info['num_classes']

                if pretrained_num_classes == self.num_class:
                    self.model.head = self._create_classification_head()
                    linear_layer = self.model.head.linear
                    with torch.no_grad():
                        linear_layer.weight.copy_(self.pretrained_head_info['weight'])
                        linear_layer.bias.copy_(self.pretrained_head_info['bias'])

                else:
                    self.model.head = self._create_classification_head()

                    if self.num_class < pretrained_num_classes:
                        linear_layer = self.model.head.linear
                        pretrained_weight = self.pretrained_head_info['weight']
                        pretrained_bias = self.pretrained_head_info['bias']

                        if pretrained_weight.shape[1] == linear_layer.weight.shape[1]:
                            with torch.no_grad():
                                linear_layer.weight.copy_(pretrained_weight[:self.num_class])
                                linear_layer.bias.copy_(pretrained_bias[:self.num_class])
                        else:
                            pass

            elif hasattr(self.model, 'head') and self.model.head is not None:
                existing_head = self.model.head

                if hasattr(existing_head, 'linear'):
                    if existing_head.linear.out_features == self.num_class:
                        pass
                    else:
                        self.model.head = self._create_classification_head()
                else:
                    last_layer = None
                    for module in existing_head.modules():
                        if isinstance(module, nn.Linear):
                            last_layer = module

                    if last_layer is not None and last_layer.out_features == self.num_class:
                        pass
                    else:
                        self.model.head = self._create_classification_head()
            else:
                input_seq_len = getattr(self.model.config, 'input_seq_len', 256)
                self.model.head = self._create_classification_head(input_seq_len)

            if self.freeze_head:
                for param in self.model.head.parameters():
                    param.requires_grad = False
            else:
                for param in self.model.head.parameters():
                    param.requires_grad = True

        else:
            raise ValueError(f"Unsupported task: {self.task_name}")
    
    def set_training_stage(self,stage):
        self.model.set_training_stage(stage)

    def set_epoch(self, epoch: int):
        """Set current epoch for GRL schedule (delegates to model)."""
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(epoch)

    def classify(self,x_enc,mask=None,subject_ids=None,labels=None,collect_features=False):
        batch_size, n_channels, seq_len = x_enc.shape

        feature_collector = None
        if collect_features and self.model.optimized_shared_factory is not None:
            feature_collector = FeatureCollector()
            self.model.optimized_shared_factory.set_feature_collector_for_all_modules(feature_collector)

        if mask is not None:
            mask = mask.to(x_enc.device)
        x_enc_norm = self.model.normalizer(x_enc, mode="norm", mask=mask)

        x_enc_norm = torch.nan_to_num(x_enc_norm, nan=0, posinf=0, neginf=0)

        x_enc_patches = self.model.tokenizer(x_enc_norm)

        if mask is None:
            mask = torch.ones((x_enc_patches.shape[0], seq_len), device=x_enc.device)

        x_enc_embedded = self.model.patch_embedding(x_enc_patches, mask=mask)

        batch_size, n_channels, n_patches, d_model = x_enc_embedded.shape
        x_enc_reshaped = x_enc_embedded.reshape(batch_size * n_channels, n_patches, d_model)

        if mask is not None:
            from momentfm.utils.masking import Masking
            patch_view_mask = Masking.convert_seq_to_patch_view(mask, self.model.patch_len)
            attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        else:
            attention_mask = None

        if subject_ids is not None and self.model.optimized_shared_factory is not None:
            self.model.optimized_shared_factory.set_subject_ids_for_all_modules(subject_ids)

        encoder_outputs = self.model.encoder(inputs_embeds=x_enc_reshaped, attention_mask=attention_mask, use_cache=False)

        last_hidden_state = encoder_outputs.last_hidden_state

        # ==========================================
        # Apply IIB (Information Invariant Bottleneck)
        # ==========================================
        if self.model.enable_iib and self.model.iib is not None:
            # Apply IIB to encoder outputs
            iib_output = self.model.iib(last_hidden_state)
            last_hidden_state = iib_output

        features = last_hidden_state.reshape(batch_size, n_channels, n_patches, d_model)

        if self.reduction == "mean":
            enc_out = features.mean(dim=1, keepdim=False)
        elif self.reduction == "concat":
            enc_out = features.permute(0, 2, 3, 1).reshape(
                batch_size, n_patches, d_model * n_channels)
        else:
            raise ValueError(f"Unsupported reduction method: {self.reduction}")

        logits = self.model.head(enc_out, input_mask=mask)

        aux_loss = self.model.get_total_aux_loss()

        # Compute IIB losses
        iib_losses = self.model.get_iib_losses(subject_ids, labels)
        iib_loss = self.model.get_total_iib_loss(subject_ids, labels)

        if collect_features and feature_collector is not None and self.model.optimized_shared_factory is not None:
            self.model.optimized_shared_factory.set_feature_collector_for_all_modules(None)

            return EnhancedClassificationOutput(
                logits=logits,
                aux_loss=aux_loss,
                gamma_params=feature_collector.gamma_params,
                beta_params=feature_collector.beta_params,
                pre_alignment_features=feature_collector.pre_alignment_features,
                post_alignment_features=feature_collector.post_alignment_features,
                layer_ids=feature_collector.layer_ids
            )
        else:
            class ClassificationOutput:
                def __init__(self, logits, aux_loss, iib_loss=None, iib_losses=None):
                    self.logits = logits
                    self.aux_loss = aux_loss
                    self.iib_loss = iib_loss  # Total weighted IIB loss
                    self.iib_losses = iib_losses  # Dictionary with individual IIB losses

            return ClassificationOutput(logits, aux_loss, iib_loss, iib_losses)
    
    def get_model_analysis(self,x_enc):
        return self.model.get_comprehensive_analysis(x_enc)
    
    def get_efficiency_report(self):
        return self.model.get_parameter_efficiency_summary()
    
    def to(self,device):
        self.model = self.model.to(device)
        return self
    
    def parameters(self):
        return self.model.parameters()
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self,state_dict,strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)
    
    def train(self,mode=True):
        self.model.train(mode)
        return self
    
    def eval(self):
        self.model.eval()
        return self
    
    @classmethod
    def from_pretrained(cls,model_path,decoupling_config=None,task_name="classification",num_class=2,model_kwargs=None,**kwargs):
        import json
        import os
        
        if model_kwargs is not None:
            kwargs.update(model_kwargs)
        
        task_name = kwargs.get('task_name', task_name)
        num_class = kwargs.get('num_class', num_class)
        freeze_pretrained = kwargs.get('freeze_pretrained', True)
        freeze_head = kwargs.get('freeze_head', True)
        reduction = kwargs.get('reduction', 'concat')
        
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        if 't5_config' in config_dict:
            t5_config = config_dict['t5_config']
            moment_config = {
                'seq_len': config_dict.get('seq_len', 512),
                'patch_len': config_dict.get('patch_len', 8),
                'patch_stride_len': config_dict.get('patch_stride_len', 8),
                'n_channels': kwargs.get('n_channels', 16),
                'task_name': config_dict.get('task_name', 'reconstruction'),
                'model_name': config_dict.get('model_name', 'MOMENT'),
                'transformer_type': config_dict.get('transformer_type', 'encoder_only'),
                'device': config_dict.get('device', 'cpu'),
                'transformer_backbone': config_dict.get('transformer_backbone', 'google/flan-t5-small'),
                'model_kwargs': config_dict.get('model_kwargs', {}),
                't5_config': t5_config,
                **t5_config
            }
        else:
            moment_config = config_dict
        
        from momentfm.utils.utils import NamespaceWithDefaults
        config = NamespaceWithDefaults(**moment_config)
        
        model = SageStream(
            config=config, 
            decoupling_config=decoupling_config
        )
        
        pretrained_head_info = None

        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            if 'head.linear.weight' in state_dict and 'head.linear.bias' in state_dict:
                pretrained_head_weight = state_dict['head.linear.weight']
                pretrained_head_bias = state_dict['head.linear.bias']
                pretrained_num_classes = pretrained_head_weight.shape[0]

                pretrained_head_info = {
                    'weight': pretrained_head_weight,
                    'bias': pretrained_head_bias,
                    'num_classes': pretrained_num_classes
                }

                if pretrained_num_classes == num_class:
                    pass
                else:
                    state_dict.pop('head.linear.weight', None)
                    state_dict.pop('head.linear.bias', None)

            model.load_state_dict(state_dict, strict=False)

        
        pipeline = cls(
            model=model,
            task_name=task_name,
            num_class=num_class,
            freeze_pretrained=freeze_pretrained,
            freeze_head=freeze_head,
            pretrained_head_info=pretrained_head_info,
            reduction=reduction
        )
        
        return pipeline
