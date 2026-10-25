from datetime import datetime

from pydantic import BaseModel


class ValidationIssue(BaseModel):
    message: str
    task_id: str | None = None
    start: datetime | None = None
    end: datetime | None = None


class ValidationResult(BaseModel):
    validator: str
    valid: bool
    issues: list[ValidationIssue] = []
