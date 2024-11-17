import gradio as gr
import warnings
import src.utils as utils
from src.predict import prediction
from src.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


def show_processing_text():
    return gr.update(visible=True), gr.update(visible=False)

def prediction_with_loading(age, gender, chest_pain, bp, cholesterol, blood_sugar, electrocardiographic, heart_rate, exercise_angina, oldpeak, slope):
    input_data_raw = f"""
    **Input Data:**
        - Age: {age}
        - Gender: {gender}
        - Chest Pain Type: {chest_pain}
        - Blood Pressure: {bp}
        - Cholesterol: {cholesterol}
        - Fasting Blood Sugar: {'Above 120 mg/dl' if blood_sugar > 120 else 'Below 120 mg/dl'}
        - Electrocardiographic Results: {electrocardiographic}
        - Maximum Heart Rate: {heart_rate}
        - Exercise Induced Angina: {exercise_angina}
        - Oldpeak: {oldpeak}
        - Slope of the Peak Exercise ST Segment: {slope}
    """
    
    gender = 1 if gender == "Male" else 0

    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
    cp = cp_map[chest_pain]

    blood_sugar = 1 if blood_sugar > 120 else 0

    electro_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    electro_result = electro_map[electrocardiographic]

    exercise_angina = 0 if exercise_angina == "No" else 1

    slope_map = {"Up-Sloping": 0, "Flat": 1, "Down-Sloping": 2}
    slope = slope_map[slope]
    
    try:
        logger.info("Starting prediction process...")
        response = prediction(
            age, gender, cp, bp, cholesterol, blood_sugar, electro_result, heart_rate, exercise_angina, oldpeak, slope, input_data_raw
        )
        logger.info("Prediction completed successfully.")
        return gr.update(value=response, visible=True), gr.update(visible=False)
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return "An error occurred during prediction. Please try again.", gr.update(visible=False)

with gr.Blocks(css=utils.css, js=utils.js, theme=gr.themes.Ocean(font=gr.themes.GoogleFont("Poppins"), primary_hue=gr.themes.colors.cyan, secondary_hue=gr.themes.colors.green), fill_width=True) as demo:
    gr.Markdown("## HEART DISEASE PREDICTION - GENAI", elem_classes="title")

    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Age", value=25)
        with gr.Column():
            gender = gr.Dropdown(choices=["Male", "Female"], label="Gender", value="Male")
        with gr.Column():
            chest_pain = gr.Dropdown(choices=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"], label="Chest Pain Type")
        with gr.Column():
            bp = gr.Number(label="Blood Pressure", value=120)
    with gr.Row():       
        with gr.Column():
            cholesterol = gr.Number(label="Cholesterol", value=200)
        with gr.Column():
            blood_sugar = gr.Number(label="Fasting Blood Sugar", value=100)
        with gr.Column():
            electrocardiographic = gr.Dropdown(choices=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"], label="Electrocardiographic Results")
        with gr.Column():
            heart_rate = gr.Number(label="Max Heart Rate", value=150)
    with gr.Row():
        with gr.Column():
            exercise_angina = gr.Dropdown(choices=["Yes", "No"], label="Exercise Induced Angina", value="No") 
        with gr.Column():
            oldpeak = gr.Number(label="Oldpeak", value=1.0, step=0.1)
        with gr.Column():       
            slope = gr.Dropdown(choices=["Up-Sloping", "Flat", "Down-Sloping"], label="Slope of the Peak Exercise ST Segment")

    with gr.Row():
        predict_button = gr.Button("Predict", variant="primary")
    
    processing_text = gr.Markdown("", visible=False, height=100)
    output_text = gr.Markdown(label="LLM Generated Diagnostic Report", container=True, show_copy_button=True, visible=False)

    predict_button.click(
        fn=show_processing_text,
        inputs=[],
        outputs=[processing_text, output_text],
        queue=False
    )
    predict_button.click(
        fn=prediction_with_loading,
        inputs=[age, gender, chest_pain, bp, cholesterol, blood_sugar, electrocardiographic, heart_rate, exercise_angina, oldpeak, slope],
        outputs=[output_text, processing_text],
        queue=True
    )

demo.launch(share=True)
