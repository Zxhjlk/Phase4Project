import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, last_conv_layer_name):
        """
        Initialize Grad-CAM for a given Keras model
        
        Args:
            model (tf.keras.Model): Trained Keras model
            last_conv_layer_name (str): Name of the last convolutional layer
        """
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        
        # Create model to get last convolutional layer output
        self.last_conv_layer = model.get_layer(last_conv_layer_name)
        
        # Create gradient model
        self.grad_model = tf.keras.models.Model(
            [model.inputs], 
            [self.last_conv_layer.output, model.output]
        )
    
    def compute_heatmap(self, img, class_index=None):
        """
        Compute Grad-CAM heatmap for a given image
        
        Args:
            img (np.array): Input image
            class_index (int, optional): Target class index. 
                                         If None, uses the predicted class.
        
        Returns:
            np.array: Grad-CAM heatmap
        """
        # Ensure image has batch dimension
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img)
            
            # If no class index specified, use the predicted class
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            
            # Compute class output for specific class
            class_output = predictions[:, class_index]
        
        # Compute gradients of class output with respect to last conv layer
        grads = tape.gradient(class_output, conv_outputs)
        
        # Global Average Pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Get conv layer output
        conv_outputs = conv_outputs[0]
        
        # Weight conv outputs by gradients
        heatmap = tf.reduce_sum(
            tf.multiply(pooled_grads, conv_outputs), 
            axis=-1
        )
        
        # ReLU to remove negative values
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap
        heatmap /= np.max(heatmap)
        
        return heatmap
    
    def overlay_heatmap(self, heatmap, img, alpha=0.4, 
                         colormap=cv2.COLORMAP_JET):
        """
        Overlay Grad-CAM heatmap on original image
        
        Args:
            heatmap (np.array): Grad-CAM heatmap
            img (np.array): Original image
            alpha (float): Transparency of heatmap
            colormap (int): OpenCV colormap to use
        
        Returns:
            np.array: Image with heatmap overlay
        """
        
        # Resize heatmap to image dimensions
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = (heatmap * 255).astype("uint8")
        
        # Apply colormap
        heatmap = cv2.applyColorMap(heatmap, colormap)
        
        # Overlay heatmap on original image
        output = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        
        return output

# Example usage
def example_usage(model, image):
    """
    Demonstrate Grad-CAM usage
    
    Args:
        model (tf.keras.Model): Trained Keras model
        image (np.array): Input image
    """
    # Create Grad-CAM instance (replace 'last_conv_layer_name' with your model's last conv layer)
    gradcam = GradCAM(model, last_conv_layer_name='last_conv_layer')
    
    # Compute heatmap
    heatmap = gradcam.compute_heatmap(image)
    
    # Optional: Overlay heatmap
    overlay = gradcam.overlay_heatmap(heatmap, image)
    
    # Visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title('Grad-CAM Heatmap')
    plt.imshow(overlay)
    plt.show()