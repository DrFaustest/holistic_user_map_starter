import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from app.models.schemas import InteractionAnalysis, PromptAssembly, UserMap


class InMemoryUserMapStore:
    def __init__(self, storage_path: Path | None = None):
        self._storage_path = storage_path or Path(__file__).resolve().parents[2] / "data" / "user_maps.db"
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def get_or_create(self, user_id: str) -> UserMap:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT map_json FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()

            if row is None:
                user_map = UserMap(user_id=user_id)
                self.save(user_map)
                return user_map

        return UserMap.model_validate_json(row[0])

    def save(self, user_map: UserMap) -> None:
        payload = user_map.model_dump_json()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO user_profiles (user_id, map_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    map_json = excluded.map_json,
                    updated_at = excluded.updated_at
                """,
                (
                    user_map.user_id,
                    payload,
                    user_map.created_at.isoformat(),
                    user_map.updated_at.isoformat(),
                ),
            )

    def record_interaction(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str,
        explicit_feedback: str | None,
        analysis: InteractionAnalysis,
    ) -> int:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO interaction_events (
                    user_id,
                    user_message,
                    assistant_response,
                    explicit_feedback,
                    analysis_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    user_message,
                    assistant_response,
                    explicit_feedback,
                    analysis.model_dump_json(),
                    datetime.utcnow().isoformat(),
                ),
            )
        return int(cursor.lastrowid)

    def get_recent_interactions(self, user_id: str, limit: int = 20) -> list[dict[str, str | None]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, user_message, assistant_response, explicit_feedback, analysis_json, created_at
                FROM interaction_events
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

        return [
            {
                "interaction_id": row[0],
                "user_message": row[1],
                "assistant_response": row[2],
                "explicit_feedback": row[3],
                "analysis_json": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]

    def get_interaction(self, interaction_id: int) -> dict[str, str | int | None] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, user_message, assistant_response, explicit_feedback, analysis_json, created_at
                FROM interaction_events
                WHERE id = ?
                """,
                (interaction_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "interaction_id": row[0],
            "user_id": row[1],
            "user_message": row[2],
            "assistant_response": row[3],
            "explicit_feedback": row[4],
            "analysis_json": row[5],
            "created_at": row[6],
        }

    def update_interaction_feedback(
        self,
        interaction_id: int,
        explicit_feedback: str,
        analysis: InteractionAnalysis,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE interaction_events
                SET explicit_feedback = ?, analysis_json = ?
                WHERE id = ?
                """,
                (explicit_feedback, analysis.model_dump_json(), interaction_id),
            )

    def get_cached_prompt_assembly(self, user_id: str, profile_version: str) -> PromptAssembly | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT prompt_json
                FROM prompt_context_cache
                WHERE user_id = ? AND profile_version = ?
                """,
                (user_id, profile_version),
            ).fetchone()

        if row is None:
            return None

        return PromptAssembly.model_validate_json(row[0])

    def save_cached_prompt_assembly(self, prompt_assembly: PromptAssembly) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO prompt_context_cache (user_id, profile_version, prompt_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, profile_version) DO UPDATE SET
                    prompt_json = excluded.prompt_json,
                    created_at = excluded.created_at
                """,
                (
                    prompt_assembly.user_id,
                    prompt_assembly.profile_version,
                    prompt_assembly.model_dump_json(),
                    datetime.utcnow().isoformat(),
                ),
            )

    def invalidate_prompt_cache(self, user_id: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM prompt_context_cache WHERE user_id = ?",
                (user_id,),
            )

    def list_user_ids(self) -> list[str]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT user_id FROM user_profiles ORDER BY user_id"
            ).fetchall()
        return [row[0] for row in rows]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._storage_path)

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    map_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS interaction_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    explicit_feedback TEXT,
                    analysis_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_interaction_events_user_id_created_at ON interaction_events(user_id, created_at DESC)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_context_cache (
                    user_id TEXT NOT NULL,
                    profile_version TEXT NOT NULL,
                    prompt_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, profile_version)
                )
                """
            )


store = InMemoryUserMapStore()
