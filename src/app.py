import streamlit as st
from PIL import Image
import tempfile
import os
from image_captioning import DualImageCaptioning

# Initialize the captioning system (cached to avoid reloading models)
@st.cache_resource
def load_captioner():
    """Load and cache the dual image captioning models"""
    return DualImageCaptioning()

def main():
    st.title("🖼️ Image Captioning with BLIP and BLIP-2 Models")
    st.write("Upload an image to generate captions using state-of-the-art vision-language models.")
    
    # Add some information about the models
    with st.expander("ℹ️ About the Models"):
        st.write("""
        - **BLIP**: Bootstrapping Language-Image Pre-training model that generates concise captions
        - **BLIP-2**: Advanced version with better understanding and more detailed descriptions
        - **Both Models**: Compare outputs from both models side by side
        """)
    
    # Model selection
    model_option = st.selectbox(
        "Choose captioning option:",
        ("BLIP Only", "BLIP-2 Only", "Both Models (Comparison)")
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG, JPEG, or PNG image file"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        raw_image = Image.open(uploaded_file).convert('RGB')
        st.image(raw_image, caption='Uploaded Image', use_column_width=True)
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            if model_option in ["BLIP Only", "Both Models (Comparison)"]:
                blip_prompt = st.text_input(
                    "BLIP Prompt:", 
                    value="a photography of",
                    help="Text prompt to guide BLIP caption generation"
                )
            
            if model_option in ["BLIP-2 Only", "Both Models (Comparison)"]:
                blip2_prompt = st.text_area(
                    "BLIP-2 Prompt:", 
                    value="Describe this image in detail with at least three sentences.",
                    help="Text prompt to guide BLIP-2 caption generation"
                )
                max_tokens = st.slider(
                    "Max New Tokens (BLIP-2):", 
                    min_value=50, 
                    max_value=500, 
                    value=200,
                    help="Maximum number of tokens for BLIP-2 generation"
                )
        
        # Generate caption button
        if st.button("🚀 Generate Caption(s)", type="primary"):
            try:
                with st.spinner('Loading models and generating captions...'):
                    # Load the captioner (cached)
                    captioner = load_captioner()
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        if model_option == "BLIP Only":
                            # Generate BLIP caption only
                            caption = captioner.generate_blip_caption(
                                tmp_file_path, 
                                text_prompt=blip_prompt if 'blip_prompt' in locals() else "a photography of"
                            )
                            st.success("Caption generated successfully!")
                            st.write("### 🔍 BLIP Caption:")
                            st.info(caption)
                            
                        elif model_option == "BLIP-2 Only":
                            # Generate BLIP-2 caption only
                            caption = captioner.generate_blip2_caption(
                                tmp_file_path, 
                                prompt=blip2_prompt if 'blip2_prompt' in locals() else "Describe this image in detail with at least three sentences.",
                                max_new_tokens=max_tokens if 'max_tokens' in locals() else 200
                            )
                            st.success("Caption generated successfully!")
                            st.write("### 🔍 BLIP-2 Caption:")
                            st.info(caption)
                            
                        else:  # Both Models (Comparison)
                            # Generate captions from both models
                            captions = captioner.generate_both_captions(
                                tmp_file_path,
                                blip_prompt=blip_prompt if 'blip_prompt' in locals() else "a photography of",
                                blip2_prompt=blip2_prompt if 'blip2_prompt' in locals() else "Describe this image in detail with at least three sentences.",
                                max_new_tokens=max_tokens if 'max_tokens' in locals() else 200
                            )
                            
                            st.success("Captions generated successfully!")
                            
                            # Display captions in columns
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("### 🔍 BLIP Caption")
                                st.info(captions['blip'])
                            
                            with col2:
                                st.write("### 🔍 BLIP-2 Caption")
                                st.info(captions['blip2'])
                            
                            # Add comparison section
                            st.write("### 📊 Comparison Analysis")
                            st.write("**BLIP** typically provides shorter, more concise captions focusing on the main subject.")
                            st.write("**BLIP-2** generates more detailed descriptions with better contextual understanding.")
                    
                    finally:
                        # Clean up temporary file
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
                            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please make sure you have the required dependencies installed:")
                st.code("pip install transformers torch pillow", language="bash")

    # Sidebar with information
    st.sidebar.title("📝 Instructions")
    st.sidebar.write("""
    1. Upload an image (JPG, JPEG, or PNG)
    2. Choose your preferred captioning model
    3. Optionally adjust advanced settings
    4. Click 'Generate Caption(s)' to see results
    """)
    
    st.sidebar.title("🔧 Requirements")
    st.sidebar.write("""
    Make sure you have installed:
    - `transformers`
    - `torch`
    - `pillow`
    - `streamlit`
    """)
    
    st.sidebar.title("💡 Tips")
    st.sidebar.write("""
    - Try different prompts for varied results
    - BLIP works well for simple descriptions
    - BLIP-2 excels at detailed analysis
    - Compare both models for best insights
    """)

if __name__ == "__main__":
    main()