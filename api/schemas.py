from pydantic import BaseModel
from datetime import datetime
from uuid import UUID

class AnalysisCreate(BaseModel):
    image_filename: str
    building_pixel_count: int = 0
    total_pixels: int = 0
    urban_percentage: int = 0
    llm_report: str = ""

class AnalysisResponse(BaseModel):
    id: UUID
    image_filename: str
    building_pixel_count: int
    total_pixels: int
    urban_percentage: int
    llm_report: str
    created_at: datetime

    class Config:
        from_attributes = True

class AnalysisHistory(BaseModel):
    analyses: list[AnalysisResponse]
    total: int
