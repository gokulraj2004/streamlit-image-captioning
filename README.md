# 🖼️ Dual Image Captioning with BLIP and BLIP-2

A comprehensive image captioning solution that leverages both BLIP and BLIP-2 models to generate detailed descriptions of images. This project provides both a programmatic interface and an interactive Streamlit web application.

## ✨ Features

- **Dual Model Support**: Compare outputs from both BLIP and BLIP-2 models
- **Interactive Web Interface**: User-friendly Streamlit application
- **Flexible Usage**: Use individual models or compare both simultaneously
- **GPU Support**: Automatic device detection with CUDA acceleration
- **Customizable Prompts**: Adjust prompts for different captioning styles
- **Easy Integration**: Clean Python class for programmatic use

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dual-image-captioning.git
cd dual-image-captioning
```

2. Install dependencies:
```bash
pip install torch torchvision transformers pillow streamlit
```

3. For GPU support (optional but recommended):
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Programmatic Usage

```python
from image_captioning import DualImageCaptioning

# Initialize the captioning system
captioner = DualImageCaptioning()

# Generate caption with BLIP
blip_caption = captioner.generate_blip_caption("path/to/your/image.jpg")

# Generate caption with BLIP-2
blip2_caption = captioner.generate_blip2_caption("path/to/your/image.jpg")

# Compare both models
captions = captioner.compare_captions("path/to/your/image.jpg")
```

## 📁 Project Structure

```
dual-image-captioning/
├── image_captioning.py    # Core captioning class with both models
├── streamlit_app.py       # Streamlit web interface
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## 🔧 Configuration Options

### BLIP Model Options
- **Text Prompt**: Customize the initial prompt (default: "a photography of")
- **Model**: Uses `Salesforce/blip-image-captioning-large`

### BLIP-2 Model Options
- **Prompt**: Detailed prompting for comprehensive descriptions
- **Max Tokens**: Control output length (default: 200)
- **Model**: Uses `Salesforce/blip2-flan-t5-xl`

## 🎯 Model Comparison

| Feature | BLIP | BLIP-2 |
|---------|------|--------|
| **Caption Style** | Concise, focused | Detailed, comprehensive |
| **Context Understanding** | Good | Excellent |
| **Memory Usage** | Lower | Higher |
| **Generation Speed** | Faster | Slower |
| **Best Use Case** | Quick descriptions | Detailed analysis |

## 📖 API Reference

### DualImageCaptioning Class

#### `__init__(device="auto")`
Initialize both BLIP and BLIP-2 models.
- `device`: Device to run models on ("auto", "cuda", "cpu")

#### `generate_blip_caption(image_path, text_prompt="a photography of")`
Generate caption using BLIP model.
- `image_path`: Path to the image file
- `text_prompt`: Text prompt to guide caption generation
- Returns: Generated caption string

#### `generate_blip2_caption(image_path, prompt="...", max_new_tokens=200)`
Generate caption using BLIP-2 model.
- `image_path`: Path to the image file
- `prompt`: Text prompt to guide caption generation
- `max_new_tokens`: Maximum number of new tokens to generate
- Returns: Generated caption string

#### `generate_both_captions(image_path, blip_prompt="...", blip2_prompt="...", max_new_tokens=200)`
Generate captions using both models.
- Returns: Dictionary with captions from both models

#### `compare_captions(image_path)`
Generate and compare captions from both models with formatted output.

## 📦 Dependencies

```txt
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.21.0
Pillow>=8.0.0
streamlit>=1.28.0
```

## 🚀 Usage Examples

### Basic Usage
```python
from image_captioning import DualImageCaptioning

captioner = DualImageCaptioning()
caption = captioner.generate_blip_caption("my_image.jpg")
print(f"Caption: {caption}")
```

### Custom Prompts
```python
# Custom BLIP prompt
blip_caption = captioner.generate_blip_caption(
    "image.jpg", 
    text_prompt="a detailed view of"
)

# Custom BLIP-2 prompt
blip2_caption = captioner.generate_blip2_caption(
    "image.jpg",
    prompt="Analyze this image and describe what you see, including colors, objects, and setting.",
    max_new_tokens=300
)
```

### Batch Processing
```python
import os

image_folder = "path/to/images"
results = {}

for image_file in os.listdir(image_folder):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, image_file)
        results[image_file] = captioner.generate_both_captions(image_path)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Salesforce Research](https://github.com/salesforce/BLIP) for the BLIP models
- [Hugging Face](https://huggingface.co/transformers/) for the Transformers library
- [Streamlit](https://streamlit.io/) for the web framework

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/dual-image-captioning/issues) page
2. Create a new issue with detailed information
3. Include your system specifications and error messages

## 🔮 Future Enhancements

- [ ] Support for batch image processing
- [ ] Additional vision-language models (LLaVA, InstructBLIP)
- [ ] Caption quality metrics and evaluation
- [ ] Export functionality for captions
- [ ] Docker containerization
- [ ] API endpoint deployment

---

Made with ❤️ by [GOKUL RAJ](https://github.com/gokulraj2004)
