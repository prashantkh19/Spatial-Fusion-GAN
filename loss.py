import torch
import torch.nn as nn

# # Lossess
# def criterion_discriminator(output, target):
#     loss = torch.mean((output - target))
#     return loss

# def criterion_generator(output):
#     loss = -1 * torch.mean(output)
#     return loss

criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

criterion_discriminator = nn.MSELoss()
criterion_generator = nn.MSELoss()