import joblib
import numpy as np
import tensorflow as tf
from keras.models import load_model  
from .llm import LLM
from .config import *
from .exception import log_exception, ModelLoadingError, PreprocessingError, PredictionError
from .logger import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self):
        try:
            self.scaler = joblib.load(f"{MODELS_DIR}/scaler_object.joblib")
            logger.info("DataPreprocessor initialized with scaler.")
        except Exception as e:
            log_exception(e, "Error loading scaler in DataPreprocessor.")
            raise ModelLoadingError("Could not load the scaler for data preprocessing.") from e

    def preprocess(self, input_data):
        try:
            scaled_data = self.scaler.transform(np.array(input_data).reshape(1, -1))
            logger.info("Data preprocessing completed successfully.")
            return scaled_data
        except Exception as e:
            log_exception(e, "Error in preprocessing data in DataPreprocessor.")
            raise PreprocessingError("Preprocessing failed. Ensure input data format is correct.") from e


class ML_Model_Predictor:
    def __init__(self):
        try:
            self.model = load_model(f"{MODELS_DIR}/dl_best_model.h5")
            logger.info("ML model loaded successfully.")
        except Exception as e:
            log_exception(e, "Failed to load ML model in ML_Model_Predictor.")
            raise ModelLoadingError("Could not load ML model. Please check model path and format.") from e

    def predict(self, preprocessed_data):
        try:
            prediction = self.model.predict(preprocessed_data)
            logger.info("ML model prediction completed successfully.")
            return prediction[0]
        except Exception as e:
            log_exception(e, "Error during ML model prediction in ML_Model_Predictor.")
            raise PredictionError("Prediction failed. Ensure input data format is correct.") from e


def prediction(age, gender, chest_pain, bp, cholesterol, blood_sugar, electrocardiographic, heart_rate, exercise_angina, oldpeak, slope, input_data_raw):
    try:
        # Initialize classes
        preprocessor = DataPreprocessor()
        ml_predictor = ML_Model_Predictor()
        llm = LLM()
        
        # Prepare structured data input for ML model
        structured_data = [age, gender, chest_pain, bp, cholesterol, blood_sugar, electrocardiographic, heart_rate, exercise_angina, oldpeak, slope]
        preprocessed_data = preprocessor.preprocess(structured_data)
        
        # Get predictions from ML and CNN models
        ml_prediction_result = np.round(ml_predictor.predict(preprocessed_data)).astype(int)

        result = f"""
        Heart Disease Diagnostic Report:
        
        **ML Model Prediction:** {'Yes' if ml_prediction_result == 1 else 'No'}
        
        {input_data_raw}
        """
        
        # Generate LLM report
        report = llm.inference(result=result)
        logger.info("LLM report generated successfully.")
        return report
    
    except Exception as e:
        log_exception(e, "Error in prediction function.")
        raise PredictionError("Prediction function encountered an error. Check inputs and model paths.") from e
