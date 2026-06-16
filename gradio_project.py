import gradio as gr

def greet(p1, p2, p3 , p4, p5, p6, p7):
    return "Hello, " + p1 + p2 + p3 + p4 +p5 + p6 + p7

project = gr.Interface(
    fn=greet,
    inputs=[gr.Textbox(label="clinical systolic blood pressure"), gr.Textbox(label="clinical DBP"),
             gr.Textbox(label="eGFR"), gr.Textbox(label="body mass index"), 
             gr.Textbox(label="nRAAs drug use"), gr.Textbox(label="history of hypertension"), gr.Textbox(label="age")],
    outputs=["text"],
    api_name="predict"
)

project.launch()