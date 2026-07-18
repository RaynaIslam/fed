import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)

from config import DEVICE
from models import get_model, get_flat_params
from data_utils import load_cifar10, dirichlet_partition, build_client_datasets
from client import local_train
from defenses import apply_defense, apply_gradient_to_model
from state_reward import evaluate_accuracy, evaluate_loss

print("Loading data...")
train_ds, test_ds = load_cifar10()
parts = dirichlet_partition(train_ds, num_clients=6, alpha=0.5, seed=42)
c_datasets = build_client_datasets(train_ds, parts, backdoor_client_ids=[0], seed=42)

from torch.utils.data import DataLoader
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

model = get_model()
print(f"Initial param norm: {get_flat_params(model).norm():.4f}")
print(f"Initial acc: {evaluate_accuracy(model, test_loader, DEVICE):.4f}")

for round_idx in range(5):
    print(f"\n--- Round {round_idx} ---")
    
    selected = list(range(3))
    gradients = {}
    losses = {}
    
    for cid in selected:
        grad, loss = local_train(
            global_model=model,
            client_dataset=c_datasets[cid],
            client_id=cid,
            is_byzantine=False,
            device=DEVICE,
        )
        gradients[cid] = grad
        losses[cid] = loss
        print(f"  Client {cid}: loss={loss:.4f}  grad_norm={grad.norm():.4f}  "
              f"grad_nan={torch.isnan(grad).any().item()}  "
              f"grad_inf={torch.isinf(grad).any().item()}")
    
    # Apply NoDefense aggregation
    agg = apply_defense(4, gradients, model, device=DEVICE)
    print(f"  Aggregated grad norm: {agg.norm():.4f}  "
          f"nan={torch.isnan(agg).any().item()}")
    
    # Check model params BEFORE update
    params_before = get_flat_params(model)
    print(f"  Params before: norm={params_before.norm():.4f}  "
          f"nan={torch.isnan(params_before).any().item()}")
    
    apply_gradient_to_model(model, agg, lr=1.0)
    
    # Check model params AFTER update
    params_after = get_flat_params(model)
    print(f"  Params after:  norm={params_after.norm():.4f}  "
          f"nan={torch.isnan(params_after).any().item()}")
    
    acc = evaluate_accuracy(model, test_loader, DEVICE)
    loss_val = evaluate_loss(model, test_loader, DEVICE)
    print(f"  Acc={acc:.4f}  Loss={loss_val:.4f}")