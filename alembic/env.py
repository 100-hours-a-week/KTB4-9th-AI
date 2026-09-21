import asyncio
from logging.config import fileConfig

from pgvector.sqlalchemy import Vector
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from src.shared.config import get_settings

# 모든 모델을 import 해야 Base.metadata 에 테이블이 등록된다.
from src.shared.model import Base  # noqa: F401

config = context.config

# alembic.ini 의 sqlalchemy.url 은 비워두고 여기서 주입한다.
config.set_main_option("sqlalchemy.url", get_settings().db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def render_item(type_, obj, autogen_context):
    """autogenerate 가 Vector 타입을 제대로 써주도록."""
    if type_ == "type" and isinstance(obj, Vector):
        autogen_context.imports.add("import pgvector.sqlalchemy")
        return f"pgvector.sqlalchemy.Vector(dim={obj.dim})"
    return False


def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_item=render_item,
        compare_type=True,  # 컬럼 타입 변경도 감지
        compare_server_default=True,
    )


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        render_item=render_item,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
