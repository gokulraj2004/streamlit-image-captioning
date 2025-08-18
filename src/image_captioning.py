from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

class DualImageCaptioning:
    def __init__(self, device="auto"):
        """
        Initialize both BLIP and BLIP-2 models
        
        Args:
            device: Device to run models on ("auto", "cuda", "cpu")
        """
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load BLIP (original) model
        print("Loading BLIP model...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        
        if self.device == "cuda":
            self.blip_model = self.blip_model.to(self.device)
        
        # Load BLIP-2 model
        print("Loading BLIP-2 model...")
        self.blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        
        if self.device == "cuda":
            self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-flan-t5-xl", 
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-flan-t5-xl"
            )
        
        print("Both models loaded successfully!")
    
    def generate_blip_caption(self, image_path, text_prompt="a photography of"):
        """
        Generate caption using BLIP model
        
        Args:
            image_path: Path to the image file
            text_prompt: Text prompt to guide caption generation
            
        Returns:
            Generated caption string
        """
        # Load the Image
        raw_image = Image.open(image_path).convert('RGB')
        
        # Prepare the Inputs
        inputs = self.blip_processor(raw_image, text_prompt, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate the Caption
        with torch.no_grad():
            output = self.blip_model.generate(**inputs)
            
        return self.blip_processor.decode(output[0], skip_special_tokens=True)
    
    def generate_blip2_caption(self, image_path, prompt="Describe this image in detail with at least three sentences.", max_new_tokens=200):
        """
        Generate caption using BLIP-2 model
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt to guide caption generation
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated caption string
        """
        # Load the Image
        raw_image = Image.open(image_path).convert("RGB")
        
        # Prepare the Inputs
        inputs = self.blip2_processor(raw_image, prompt, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device, torch.float16) for k, v in inputs.items()}
        
        # Generate the Caption
        with torch.no_grad():
            output = self.blip2_model.generate(**inputs, max_new_tokens=max_new_tokens)
            
        return self.blip2_processor.decode(output[0], skip_special_tokens=True)
    
    def generate_both_captions(self, image_path, blip_prompt="a photography of", blip2_prompt="Describe this image in detail with at least three sentences.", max_new_tokens=200):
        """
        Generate captions using both models
        
        Args:
            image_path: Path to the image file
            blip_prompt: Prompt for BLIP model
            blip2_prompt: Prompt for BLIP-2 model
            max_new_tokens: Maximum tokens for BLIP-2
            
        Returns:
            Dictionary with captions from both models
        """
        results = {}
        
        print("Generating BLIP caption...")
        results['blip'] = self.generate_blip_caption(image_path, blip_prompt)
        
        print("Generating BLIP-2 caption...")
        results['blip2'] = self.generate_blip2_caption(image_path, blip2_prompt, max_new_tokens)
        
        return results
    
    def compare_captions(self, image_path):
        """
        Generate and compare captions from both models
        
        Args:
            image_path: Path to the image file
        """
        captions = self.generate_both_captions(image_path)
        
        print("\n" + "="*50)
        print("CAPTION COMPARISON")
        print("="*50)
        print(f"\nBLIP Caption:")
        print("-" * 20)
        print(captions['blip'])
        
        print(f"\nBLIP-2 Caption:")
        print("-" * 20)
        print(captions['blip2'])
        print("\n" + "="*50)
        
        return captions


# Example usage
if __name__ == "__main__":
    # Initialize the dual captioning system
    captioner = DualImageCaptioning()
    
    # Example image path - replace with your image path
    image_path = "/content/image_1.jpg"
    
    # Method 1: Generate individual captions
    blip_caption = captioner.generate_blip_caption(image_path)
    print(f"BLIP: {blip_caption}")
    
    blip2_caption = captioner.generate_blip2_caption(image_path)
    print(f"BLIP-2: {blip2_caption}")
    
    # Method 2: Generate both captions at once
    both_captions = captioner.generate_both_captions(image_path)
    
    # Method 3: Generate and compare captions with formatted output
    captioner.compare_captions(image_path)