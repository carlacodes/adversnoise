
import torch
import ast
import urllib.request
from PIL import ImageDraw
from PIL import ImageFont
import urllib.request
from PIL import Image
from torchvision import transforms
from helpers import deepfool

def call_torch_image_classification():
    # Load the pretrained model
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.eval()
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
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

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

def add_adversarial_noise():
    # Load the pretrained model
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.eval()
    # Download an example image from the pytorch website
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try: urllib.request.urlretrieve(url, filename)
    except Exception as e: print(e)
    # Open the image file
    input_image = Image.open(filename)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the image
    input_tensor = preprocess(input_image)
    r, loop_i, label_orig, label_pert, pert_image = deepfool.deepfool(input_tensor, model)

    return pert_image
    #next use deepfool implementation to add some adversarial noise, this is from a 2016 CPVR paper which improves on the FGSM attack, https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.pdf


    # Define the preprocessing transformation

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    add_adversarial_noise()
    call_torch_image_classification()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
