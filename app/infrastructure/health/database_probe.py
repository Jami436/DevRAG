from sqlalchemy import Engine, text

from app.domain.health.entities import DependencyStatus


class DatabaseProbe:
    """Check that the database engine can serve a trivial query."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def check(self) -> DependencyStatus:
        try:
            with self._engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception as exc:
            return DependencyStatus(
                name="database",
                state="error",
                detail=f"{type(exc).__name__}: {exc}",
            )
        return DependencyStatus(name="database", state="ok")