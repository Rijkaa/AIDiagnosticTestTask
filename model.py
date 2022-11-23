import segmentation_models_pytorch as smp
import torch
from torch.optim import lr_scheduler


def model_act():
    loss_fn = smp.losses.DiceLoss(mode='multiclass', from_logits=True)

    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                      # model output channels (number of classes in your dataset)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    lr_scheduler_exp = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    # move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model, device, optimizer, loss_fn, lr_scheduler_exp
