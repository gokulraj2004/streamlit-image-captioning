# Streamlit Image Captioning

This project implements an image captioning application using the BLIP model. It allows users to upload images and generates descriptive captions based on the content of the images.

## Project Structure

```
streamlit-image-captioning
├── src
│   ├── image_captioning.py  # Contains the image captioning functionality using the BLIP model.
│   └── app.py               # Main entry point for the Streamlit application.
├── requirements.txt          # Lists the dependencies required for the project.
└── README.md                 # Documentation for the project.
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd streamlit-image-captioning
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Upload an image using the provided interface, and the application will generate a caption for the image.

## Dependencies

- Streamlit
- Pillow
- Transformers

## License

This project is licensed under the MIT License.