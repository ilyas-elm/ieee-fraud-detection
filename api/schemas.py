from pydantic import BaseModel
from typing import Optional

class TransactionRequest(BaseModel):
    TransactionAmt: float
    ProductCD: Optional[str] = None
    card1: Optional[int] = None
    card4: Optional[str] = None
    card6: Optional[str] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    TransactionDT: Optional[int] = 0

class PredictResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold_used: float
    model_version: str
    inference_time_ms: float

class ShapFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str

class ExplainResponse(BaseModel):
    fraud_probability: float
    model_version: str
    top_features: list[ShapFeature]