"""
Grad-CAM for the EfficientNet-B3 classifier.

Produces a heatmap showing which regions of the image drove the
prediction. For a dermatology model this matters more than usual: it
is the difference between a model that has learned lesion morphology
and one that has learned to recognise ruler marks, ink annotations or
skin tone in the training images.

Usage:
    from model import build_model
    from grad_cam import GradCAM, overlay_heatmap

    model = build_model(num_classes=len(CLASSES))
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))

    cam = GradCAM(model, model.features[-1])
    heatmap, predicted_idx = cam(input_tensor)
    cam.remove_hooks()

    overlay = overlay_heatmap('lesion.jpg', heatmap)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Gradient-weighted Class Activation Mapping via forward/backward hooks."""

    def __init__(self, model, target_layer):
        """
        Args:
            model:        a trained SkinDiseaseModel in eval mode
            target_layer: the conv layer to visualise. For this
                          architecture, model.features[-1] — the last
                          convolutional block before global pooling.
        """
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        """
        Args:
            input_tensor: preprocessed image, shape (1, 3, H, W)
            class_idx:    class to explain. Defaults to the predicted class.

        Returns:
            (heatmap, class_idx) where heatmap is a HxW array in [0, 1]
        """
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Average the gradients over spatial dimensions to get a weight
        # per channel: how much each feature map contributed to this class.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU: only regions with a positive influence on this class matter.
        # Negative values indicate evidence against it.
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx

    def remove_hooks(self):
        """Detach the hooks. Call this when finished, or they leak."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def overlay_heatmap(img_path, heatmap, alpha=0.4):
    """
    Superimpose a Grad-CAM heatmap on the original image.

    Returns an RGB uint8 array suitable for matplotlib or PIL.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = heatmap * alpha + img * (1 - alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    print(__doc__)
