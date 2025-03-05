import torch
from PIL import Image
from torchvision import models, transforms

# [
#     "ResNeXt101_32X8D_Weights",
#     "ResNeXt101_64X4D_Weights",
#     "ResNeXt50_32X4D_Weights",
#     "ResNet",
#     "ResNet101_Weights",
#     "ResNet152_Weights",
#     "ResNet18_Weights",
#     "ResNet34_Weights",
#     "ResNet50_Weights",
#     "Wide_ResNet101_2_Weights",
#     "Wide_ResNet50_2_Weights",
#     "resnet",
#     "resnet101",
#     "resnet152",
#     "resnet18",
#     "resnet34",
#     "resnet50",
#     "resnext101_32x8d",
#     "resnext101_64x4d",
#     "resnext50_32x4d",
#     "wide_resnet101_2",
#     "wide_resnet50_2",
# ]


model = models.resnet101(pretrained=True, progress=True)
categories = models.ResNet101_Weights.DEFAULT.meta["categories"]
model.eval()

print(sum(p.numel() for p in model.parameters()))

input_image = Image.open("data/train/5/00e0f23c-1dfd-4032-a9bb-d5f0c0e346d6.jpg")
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to("cuda")
    model.to("cuda")

with torch.no_grad():
    output = model(input_batch)

# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities.shape)
_, index = torch.max(probabilities, dim=0)
print(categories[index])
