
import torch
import ast
import urllib.request
from PIL import ImageDraw
from PIL import ImageFont
import urllib.request
from PIL import Image
from torchvision import transforms
from helpers import deepfool

#TODO: add docstrings, make into class structure for better organization and reusability
def call_torch_image_classification(test_image = None):
    # Load the pretrained model
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Download an example image from the pytorch website
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try: urllib.request.urlretrieve(url, filename)
    except Exception as e: print(e)
    # Open the image file
    input_image = Image.open(filename)
    #display the input image for fun
    input_image.show()

    # Define the preprocessing transformation
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the image
    input_tensor = preprocess(input_image)


    if test_image is not None:
        input_tensor = test_image

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


    draw = ImageDraw.Draw(input_image)
    font = ImageFont.load_default()
    draw.text((10, 10), f"Prediction: {category}", fill="white", font=font, font_size=20)  # Specify text color
    input_image.show()
    return top5_prob, top5_catid

def add_adversarial_noise(input_image = None, target_label = None):
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

    # Preprocess the image
    input_tensor = preprocess(input_image)
    r, loop_i, label_orig, label_pert, pert_image = deepfool.deepfool_mod(input_tensor, model, target_label)
    #visualise the pert_image, convert to PIL image
    pert_image_vis = transforms.ToPILImage()(pert_image.squeeze(0))
    pert_image_vis.show()

    # Define the mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # Reverse the normalization
    input_tensor_rev = input_tensor.clone()  # Create a copy to avoid changing the original tensor
    input_tensor_rev = input_tensor_rev.squeeze(0)  # Remove the batch dimension
    input_tensor_rev = input_tensor_rev * std + mean  # Reverse the normalization

    # Convert to a PIL image
    input_tensor_rev = input_tensor_rev.squeeze(0)

    # Now convert to a PIL image
    input_tensor_vis = transforms.ToPILImage()(input_tensor_rev)
    input_tensor_vis.show()
    return pert_image
    #next use deepfool implementation to add some adversarial noise, this is from a 2016 CPVR paper which improves on the FGSM attack, by finding the minimum perturbation needed to fool a traditional resnet model, https://github.com/LTS4/DeepFool, https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.pdf


    # Define the preprocessing transformation

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pert_image = add_adversarial_noise(target_label='zebra')
    call_torch_image_classification(test_image = pert_image)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
