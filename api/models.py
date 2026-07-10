import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Text, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from api.database import Base

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_filename = Column(String(255), nullable=False)
    building_pixel_count = Column(Integer, default=0)
    total_pixels = Column(Integer, default=0)
    urban_percentage = Column(Integer, default=0)
    llm_report = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON, default={})
