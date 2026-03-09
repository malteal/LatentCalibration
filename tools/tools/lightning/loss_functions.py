# define pytorch loss functions

import torch.nn as nn

def get_loss(name: str, args: dict = {}):
	loss_functions = {
		'mse': nn.MSELoss,
		'mae': nn.L1Loss,
		'cross_entropy': nn.CrossEntropyLoss,
		'nll': nn.NLLLoss,
		'bce': nn.BCELoss,
		'bce_with_logits': nn.BCEWithLogitsLoss,
		'hinge': nn.HingeEmbeddingLoss,
		'kl_div': nn.KLDivLoss,
		'smooth_l1': nn.SmoothL1Loss,
		'huber': nn.HuberLoss,
		'cosine_embedding': nn.CosineEmbeddingLoss,
		'margin_ranking': nn.MarginRankingLoss,
		'multi_margin': nn.MultiMarginLoss,
		'multi_label_margin': nn.MultiLabelMarginLoss,
		'soft_margin': nn.SoftMarginLoss,
		'multi_label_soft_margin': nn.MultiLabelSoftMarginLoss,
		'triplet_margin': nn.TripletMarginLoss,
		'ctc': nn.CTCLoss
	}

	if name not in loss_functions:
		raise ValueError(f"Loss function '{name}' is not supported.")

	return loss_functions[name](**args)
    