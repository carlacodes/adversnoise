import unittest
from torchvision.transforms import ToTensor
from PIL import Image
import torch
from run_advnoise import ImageClassifier, AdversarialNoiseGenerator


class TestImageClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = ImageClassifier()

    def test_classify(self):


        # Call the classify method
        top5_prob, top5_catid = self.classifier.classify(pert_image, visualise_output=False)

        # Check the output types
        self.assertIsInstance(top5_prob, torch.Tensor)
        self.assertIsInstance(top5_catid, torch.Tensor)

        # Check the output shapes
        self.assertEqual(top5_prob.shape, (5,))
        self.assertEqual(top5_catid.shape, (5,))

class TestAdversarialNoiseGenerator(unittest.TestCase):
    def setUp(self):
        self.noise_generator = AdversarialNoiseGenerator()

    def test_add_noise(self):
        # Load a test image and convert it to a tensor
        image = Image.open("../images/dog.jpg")

        # Call the add_noise method
        pert_image = self.noise_generator.add_noise(image, label_str='zebra', sanity_check_vis=False)

        # Check the output type
        self.assertIsInstance(pert_image, torch.Tensor)


if __name__ == '__main__':
    unittest.main()