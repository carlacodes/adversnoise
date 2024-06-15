import torch
import ast
import urllib.request
from PIL import ImageFont, Image, ImageDraw
from torchvision import transforms
from helpers import algorithms
import numpy as np

class ImageClassifier:
    """
    A class used to classify images using a pretrained PyTorch model.

    Methods
    -------
    classify(input_tensor, visualise_output=True)
        Classify an image using a pretrained PyTorch model.
    """
    def __init__(self, model_name='resnet18', device=None):
        self.model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = self.model.to(self.device)

    def classify(self, input_tensor, visualise_output=True):
        """
        Classify an image using a pretrained PyTorch model.

        Parameters
        ----------
        input_tensor : torch.Tensor
            the input image tensor
        visualise_output : bool, optional
            whether to visualise the top 5 predictions (default is True)

        Returns
        -------
        top5_prob : torch.Tensor
            the top 5 probabilities
        top5_catid : torch.Tensor
            the top 5 category IDs
        """
        if len(input_tensor.shape) != 4:
            input_batch = input_tensor.unsqueeze(0)
        else:
            input_batch = input_tensor
        input_batch = input_batch.to(self.device)

        with torch.no_grad():
            output = self.model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        _, top_catid = torch.max(probabilities, 0)

        url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
        with urllib.request.urlopen(url) as response:
            categories = ast.literal_eval(response.read().decode())

        top_catid_item = top_catid.item()
        category = categories.get(top_catid_item, "Unknown")

        print(f"The model classified the image as: {category}")
        if visualise_output:
            print("Top 5 predictions:")
            for i in range(top5_prob.size(0)):
                cat_id = top5_catid[i].item()
                cat_prob = top5_prob[i].item()
                print(f"{categories.get(cat_id, 'Unknown')}: {cat_prob:.5f}")
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            mean = mean.to(self.device)
            std = std.to(self.device)
            input_tensor_rev = input_tensor.clone()  # Create a copy to avoid changing the original tensor
            input_tensor_rev = input_tensor_rev.squeeze(0)  # Remove the batch dimension
            input_tensor_rev = input_tensor_rev * std + mean  # Reverse the normalization
            input_tensor_rev = input_tensor_rev.squeeze(0)
            input_tensor_vis = transforms.ToPILImage()(input_tensor_rev)
            draw = ImageDraw.Draw(input_tensor_vis)
            font = ImageFont.load_default()
            draw.text((10, 10), f"Prediction: {category}, \n probability: {top5_prob[0]}", fill="blue", font=font, font_size=20)  # Specify text color
            input_tensor_vis.show()

        return top5_prob, top5_catid

class AdversarialNoiseGenerator:
    """
     A class used to generate adversarial noise on images.

     Attributes
     ----------
     model : torch model
         a pretrained PyTorch model
     device : str
         the device to run the model on ('cpu' or 'cuda')

     Methods
     -------
     add_noise(input_image=None, label_str=None, sanity_check_vis=False)
         Adds adversarial noise to an image.
     """
    def __init__(self, model_name='resnet18', device=None):
        self.model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = self.model.to(self.device)

    def add_noise(self, input_image=None, label_str=None, sanity_check_vis=False):
        """
        Adds adversarial noise to an image in a supervised target manner using the PGD method.

        Parameters
        ----------
        input_image : PIL Image or None, optional
            the input image (default is None, which means a default image is used)
        label_str : str or None, optional
            the target label for the adversarial attack (default is None)
        sanity_check_vis : bool, optional
            whether to visualize the original and perturbed (pert) images (default is False)

        Returns
        -------
        pert_image : torch.Tensor
            the perturbed image
        """
        if label_str is None or not isinstance(label_str, str):
            print("Please provide a valid label string")
            return

        if input_image is None:
            url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
            try: urllib.request.urlretrieve(url, filename)
            except Exception as e: print(e)
            input_image = Image.open(filename)
        elif not isinstance(input_image, Image.Image):
            print("Input image is not a PIL image..will try to convert")
            try:
                input_image = Image.open(input_image)
            except Exception as e:
                print("Could not convert input image to PIL image.m {}".format(e))
                return

        url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
        with urllib.request.urlopen(url) as response:
            categories = ast.literal_eval(response.read().decode())

        target_label = None
        for key, value in categories.items():
            if value == label_str:
                target_label = key
                break

        if target_label is None:
            print(f"Label {label_str} not found in categories.")
            return

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_tensor = input_tensor.unsqueeze(0)
        target_labels = torch.full((input_tensor.size(0),), target_label, dtype=torch.long)
        input_tensor = input_tensor.to(self.device)
        pert_image = algorithms.pgd_attack(self.model, input_tensor, target_labels, eps=0.1, alpha=0.01, iters=100)

        if sanity_check_vis:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            mean = mean.to(self.device)
            std = std.to(self.device)
            input_tensor_rev = input_tensor.clone()  # Create a copy to avoid changing the original tensor
            input_tensor_rev = input_tensor_rev.squeeze(0)  # Remove the batch dimension
            input_tensor_rev = input_tensor_rev * std + mean  # Reverse the normalization
            input_tensor_rev = input_tensor_rev.squeeze(0)
            input_tensor_vis = transforms.ToPILImage()(input_tensor_rev)
            pert_image_rev = pert_image.clone()


            # device = pert_image_rev.device
            # mean = mean.to(device)
            # std = std.to(device)

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
    noise_generator = AdversarialNoiseGenerator()
    # dog_image = Image.open("images/dog.jpg")
    #
    # ##input path to the image you want to clasify and the label_str you want to classify it as
    # pert_image = noise_generator.add_noise(input_image=dog_image, label_str='zebra', sanity_check_vis=True)

    image_path = input("Please enter the path to the image (press Enter to use the default image): ")
    if not image_path:
        image_path = 'images/dog.jpg'
    label_str = input("Please enter the label string: ")
    sanity_check_vis = input("Do you want to perform a sanity check \n visualisation of the image before and after perturbation? (yes/no): ")

    # Convert the sanity_check_vis input to a boolean
    sanity_check_vis = True if sanity_check_vis.lower() == 'yes' else False

    # Open the image using the provided path
    input_image = Image.open(image_path)

    pert_image = noise_generator.add_noise(input_image=input_image, label_str=label_str,
                                           sanity_check_vis=sanity_check_vis)



    classifier = ImageClassifier()
    classifier.classify(pert_image)