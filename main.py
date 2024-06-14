# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


import urllib.request
from PIL import Image
from torchvision import transforms

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

    # Check if the top prediction is a dog
    #find the corresponding category for top_catid
    #https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

    dog_categories = list(range(151, 269))

    if top_catid.item() in dog_categories:
        print("The model correctly classified the image as a dog.")
    else:
        print("The model did not classify the image as a dog.")

    #display the same image with the classification displayed on top
    from PIL import ImageDraw
    from PIL import ImageFont
    draw = ImageDraw.Draw(input_image)
    font = ImageFont.load_default()
    draw.text((10, 10), f"Prediction: {top5_catid[0].item()}", font=font)
    input_image.show()



    return top5_prob, top5_catid

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    call_torch_image_classification()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/