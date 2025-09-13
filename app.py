import gradio as gr

def say_hello(name):
    return f"Hello {name}, the model is ready on Hugging Face Spaces!"

demo = gr.Interface(
    fn=say_hello,
    inputs="text",
    outputs="text",
    title="Test Space"
)

if __name__ == "__main__":
    demo.launch()
