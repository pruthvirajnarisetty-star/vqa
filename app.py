import gradio as gr
from main import final_pipeline  # Changed from utils to main
from PIL import Image
import torch

CSS = """
.gradio-container {
    max-width: 1200px !important;
}
"""

def predict(image, question):
    if image is None:
        return "Please upload an image! 📸"
    if not question.strip():
        return "Please enter a question! ❓"
    
    try:
        result = final_pipeline(image, question)
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"

with gr.Blocks(css=CSS, title="🩻 Medical VQA - MultiLanguage") as demo:
    gr.Markdown("""
    # 🩻 **Medical VQA - MultiLanguage**
    Upload X-ray, MRI, CT scans and ask questions in **ANY language**!
    
    **Supports:** English, Hindi, Tamil, Telugu, Kannada, Malayalam
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Medical Image")
            question_input = gr.Textbox(
                label="Question (Any Language)", 
                placeholder="What is shown in this X-ray? / இதில் என்ன உள்ளது?",
                lines=2
            )
            predict_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            answer_output = gr.Markdown(label="🤖 AI Answer")
    
    predict_btn.click(predict, [image_input, question_input], answer_output)
    
    gr.Examples(
        examples=[
            ["What is this fracture?", None],
            ["இதில் எந்த உடல்நலக் குறைபாடு உள்ளது?", None],
            ["ఈ చిత్రంలో ఏముంది?", None],
            ["Describe this X-ray", None]
        ],
        inputs=[question_input, image_input],
        outputs=answer_output
    )
    
    gr.Markdown("---")
    gr.Markdown("""
    **Powered by:** Custom VQA (ResNet18+LSTM) + BLIP2 + NLLB-200  
    **Dataset:** VQA-RAD (Medical Images)
    """)

if __name__ == "__main__":
    demo.launch()
