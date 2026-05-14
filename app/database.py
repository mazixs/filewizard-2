from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, field_serializer
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, event, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

from app.config import PATHS

# --- Database Engine ---
engine = create_engine(
    PATHS.DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    poolclass=NullPool,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, _connection_record):
    c = dbapi_connection.cursor()
    try:
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
    finally:
        c.close()


# --- SQLAlchemy Models ---
class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=True)
    parent_job_id = Column(String, index=True, nullable=True)
    task_type = Column(String, index=True)
    status = Column(String, default="pending")
    progress = Column(Integer, default=0)
    original_filename = Column(String)
    input_filepath = Column(String)
    input_filesize = Column(Integer, nullable=True)
    processed_filepath = Column(String, nullable=True)
    output_filesize = Column(Integer, nullable=True)
    result_preview = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    callback_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Pydantic Schemas ---
class JobCreate(BaseModel):
    id: str
    user_id: str | None = None
    parent_job_id: str | None = None
    task_type: str
    original_filename: str
    input_filepath: str
    input_filesize: int | None = None
    callback_url: str | None = None
    processed_filepath: str | None = None


class JobSchema(BaseModel):
    id: str
    parent_job_id: str | None = None
    task_type: str
    status: str
    progress: int
    original_filename: str
    input_filesize: int | None = None
    output_filesize: int | None = None
    processed_filepath: str | None = None
    result_preview: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)

    @field_serializer("created_at", "updated_at")
    def serialize_dt(self, dt: datetime, _info):
        return dt.isoformat() + "Z"


class FinalizeUploadPayload(BaseModel):
    upload_id: str
    original_filename: str
    total_chunks: int
    task_type: str
    model_size: str = ""
    model_name: str = ""
    output_format: str = ""
    generate_timestamps: bool = False
    callback_url: str | None = None


class JobSelection(BaseModel):
    job_ids: list[str]


class JobStatusRequest(BaseModel):
    job_ids: list[str]
