
import torch
import ast
import urllib.request
from PIL import ImageFont
import urllib.request
from PIL import Image
from torchvision import transforms
from helpers import deepfool
from PIL import ImageDraw
import numpy as np

#TODO: add docstrings, make into class structure for better organization and reusability
def call_torch_image_classification(input_tensor):
    '''This function takes an input tensor and classifies it using a pretrained resnet18 model from pytorch
    args:
        input_tensor: torch.tensor: the input tensor to classify
    returns:
        top5_prob: torch.tensor: the top 5 probabilities given from the pretrained resnet18 model
        '''
    # Load the pretrained model
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if len(input_tensor.shape) == 4:
        input_batch = input_tensor
    else:
        input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)

        # Make the prediction
    with torch.no_grad():
        output = model(input_batch)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Print the top 5 probabilities
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(f"Probability: {top5_prob[i].item()} Category ID: {top5_catid[i].item()}")
    _, top_catid = torch.max(probabilities, 0)

    url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    with urllib.request.urlopen(url) as response:
        categories = ast.literal_eval(response.read().decode())

    # Find the corresponding category for top_catid
    top_catid_item = top_catid.item()

    top_catid_item = top_catid.item()
    if top_catid_item in categories:
        category = categories[top_catid_item]
    else:
        category = "Unknown"

    print(f"The model classified the image as: {category}")


    draw = ImageDraw.Draw(input_tensor)
    font = ImageFont.load_default()
    draw.text((10, 10), f"Prediction: {category}", fill="white", font=font, font_size=20)  # Specify text color
    input_tensor.show()
    return top5_prob, top5_catid

def add_adversarial_noise(input_image = None, target_label = None, sanity_check_vis = False):
    '''This function takes an input image and adds adversarial noise to it using the PGD algorithm
    args:
        input_image: PIL image: the input image to add adversarial noise to
        target_label: str: the target label to add adversarial noise to
        sanity_check_vis: bool: whether to visualize the adversarial noise
    returns:
        pert_image: torch.tensor: the perturbed image with adversarial noise added
        '''
    # Load the pretrained model
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.eval()
    if input_image is None:
        # Download an example image from the pytorch website
        url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        try: urllib.request.urlretrieve(url, filename)
        except Exception as e: print(e)
        # Open the image file
        input_image = Image.open(filename)
    try:
        assert isinstance(input_image, Image.Image)
    except AssertionError as e:
        print(e)
        return


    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_tensor = input_tensor.unsqueeze(0)
    target_label = 358  # The target label
    target_labels = torch.full((input_tensor.size(0),), target_label, dtype=torch.long)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    pert_image = deepfool.pgd_attack(model, input_tensor, target_labels, eps=0.3, alpha=0.01, iters=100)


    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # Reverse the normalization
    if sanity_check_vis:
        input_tensor_rev = input_tensor.clone()  # Create a copy to avoid changing the original tensor
        input_tensor_rev = input_tensor_rev.squeeze(0)  # Remove the batch dimension
        input_tensor_rev = input_tensor_rev * std + mean  # Reverse the normalization
        input_tensor_rev = input_tensor_rev.squeeze(0)
        input_tensor_vis = transforms.ToPILImage()(input_tensor_rev)
        pert_image_rev = pert_image.clone()
        device = pert_image_rev.device
        mean = mean.to(device)
        std = std.to(device)

        # Create a copy to avoid changing the original tensor
        pert_image_rev = pert_image_rev.squeeze(0)  # Remove the batch dimension
        pert_image_rev = pert_image_rev * std + mean  # Reverse the normalization
        # Convert to a PIL image
        pert_image_rev = pert_image_rev.squeeze(0)
        # Now convert to a PIL image
        pert_image_vis = transforms.ToPILImage()(pert_image_rev)

        # Assuming input_tensor_vis and pert_image_vis are PIL Images
        draw_input = ImageDraw.Draw(input_tensor_vis)
        draw_pert = ImageDraw.Draw(pert_image_vis)

        # Add text
        draw_input.text((10, 10), f"Before", fill="black")
        draw_pert.text((10, 10), f"After: {target_label}", fill="black")

        # Concatenate the two images
        combined = Image.fromarray(np.concatenate((np.array(input_tensor_vis), np.array(pert_image_vis)), axis=1))
        combined.show()

    return pert_image


if __name__ == '__main__':
    pert_image = add_adversarial_noise(target_label='zebra')
    call_torch_image_classification(pert_image)

