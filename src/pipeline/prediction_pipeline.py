import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Cement:float,
                 Blast_Furnace:float,
                 Fly_Ash:float,
                 Water:float,
                 Superplasticizer:float,
                 Coarse_Aggregate:float,
                 Fine_Aggregate	:float,
                 Age:float,
                 Concrete_compressive_strength:float
                 ):
        
        self.Cement=Cement
        self.Blast_Furnace=Blast_Furnace
        self.Fly_Ash=Fly_Ash
        self.Water=Water
        self.Superplasticizer=Superplasticizer
        self.Coarse_Aggregate=Coarse_Aggregate
        self.Fine_Aggregate=Fine_Aggregate
        self.Age = Age
        self.Concrete_compressive_strength=Concrete_compressive_strength

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Cement':[self.Cement],
                'Blast_Furnace':[self.Blast_Furnace],
                'Fly_Ash':[self.Fly_Ash],
                'Water':[self.Water],
                'Superplasticizer':[self.Superplasticizer],
                'Coarse_Aggregate':[self.Coarse_Aggregate],
                'Fine_Aggregate':[self.Fine_Aggregate],
                'Age':[self.Age],
                'Concrete_compressive_strength':[self.Concrete_compressive_strength]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
