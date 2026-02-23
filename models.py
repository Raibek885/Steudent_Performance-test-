from pydantic import BaseModel
from typing import Literal

class ModelFeature(BaseModel):
    """Input features for the model."""

    Hours_Studied: float
    Attendance: float
    Sleep_Hours: float
    Previous_Scores: float
    Tutoring_Sessions: float
    Physical_Activity: float


    Parental_Involvement: Literal["Low", "Medium", "High"]
    Access_to_Resources: Literal["Low", "Medium", "High"]
    Extracurricular_Activities: Literal["No", "Yes"]
    Motivation_Level: Literal["Low", "Medium", "High"]
    Internet_Access: Literal["No", "Yes"]
    Family_Income: Literal["Low", "Medium", "High"]
    Teacher_Quality: Literal["Low", "Medium", "High"]
    School_Type: Literal["Public", "Private"]
    Peer_Influence: Literal["Negative", "Neutral", "Positive"]
    Learning_Disabilities: Literal["No", "Yes"]
    Parental_Education_Level: Literal["High School", "College", "Postgraduate"]
    Distance_from_Home: Literal["Near", "Moderate", "Far"]
    Gender: Literal["Male", "Female"]

class PredictionOut(BaseModel):
    predicted_value: float