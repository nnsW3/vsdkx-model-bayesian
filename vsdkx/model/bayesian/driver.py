import torch
from vsdkx.core.interfaces import ModelDriver
from vsdkx.core.structs import Inference

from vsdkx.model.bayesian.vgg import VGG, make_layers
from torchvision import transforms


class BayesianDriver(ModelDriver):
    """
    Class for Crowd number estimator
    """

    def __init__(self, model_settings: dict, model_config: dict,
                 drawing_config: dict):
        """
        Args:
            model_config (dict): Config dictionary with the following keys:
                'model_path' (str): Path to torch model
                'mean' (array): Array with mean values for pixel norm
                'std' (array): Array with standard deviation values
                for pixel norm
            model_settings (dict): Model settings config with the following keys:
                'device' (string): Device name ('cpu', 'gpu')
        """

        super().__init__(model_settings, model_config, drawing_config)
        self._device = model_settings['device']
        self._mean = model_config['mean']
        self._std = model_config['std']
        self._cfg = model_config['cfg']
        # Load the model
        self._model = VGG(make_layers(self._cfg['E']))
        self._model.load_state_dict(torch.load(model_config['model_path'],
                                               map_location=torch.device(
                                                  self._device)))

    def inference(self, image) -> Inference:
        """
        Inferences the input image

        Args:
            image (np.array): 3D image array

        Returns:
            (float): Crowd number estimation
        """
        # Resize the image
        image = self._resize_image(image)

        # Run the inference on the input image
        with torch.set_grad_enabled(False):
            output = self._model(image)
            crowd_no = torch.sum(output).item()

        return Inference(extra={"crowd_number": crowd_no})

    def _resize_image(self, image):
        """
        Resizes the image for inference

        Args:
            image (np.array): 3D image array

        Returns:
            (tensor): Resized image tensor
        """
        # Init the transformer
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self._mean, self._std)
        ])

        # Transform the image
        image = transform(image)
        # Expand the image dimensions
        image = torch.unsqueeze(image, 0)

        return image
