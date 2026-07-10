import os
from urllib.parse import urlparse, urlencode, urlunparse, parse_qs
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv("DATABASE_URL", "")

if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    parsed = list(urlparse(DATABASE_URL))
    params = parse_qs(parsed[4])  # query params
    params.pop("sslmode", None)
    params.pop("channel_binding", None)
    parsed[4] = urlencode(params, doseq=True)
    clean_url = urlunparse(parsed)
    DATABASE_URL = clean_url.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    connect_args={
        "prepared_statement_cache_size": 0,
        "ssl": "require",
    },
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    try:
        async with engine.begin() as conn:
            from api.models import Analysis
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        import logging
        logging.warning(f"Database init skipped: {e}")
