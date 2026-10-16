"""Alembic environment: runs migrations on the chat history database."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from app.core.chat_store import metadata
from app.core.config import settings

config = context.config
target_metadata = metadata  # what `alembic revision --autogenerate` compares against


def run_on(connection):
    context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


connection = config.attributes.get("connection")
if connection is not None:
    # Called from the app (chat_store.init_db) with its own connection: leave the
    # app's logging configuration alone.
    run_on(connection)
elif context.is_offline_mode():
    context.configure(url=settings.database_url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    # `alembic ...` on the command line, from backend/.
    if config.config_file_name:
        fileConfig(config.config_file_name)
    engine = create_engine(settings.database_url)
    with engine.begin() as conn:
        run_on(conn)
    engine.dispose()
