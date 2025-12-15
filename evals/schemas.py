from pydantic import BaseModel
from typing import Literal


class AnswerEval(BaseModel):
    grounded: Literal["yes", "no"]
    hallucination: Literal["yes", "no"]
    pii_leak: Literal["yes", "no"]
    correct: Literal["yes", "partial", "no"]
    reason: str
