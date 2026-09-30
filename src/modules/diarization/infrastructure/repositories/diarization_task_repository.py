import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, text
from sqlalchemy.orm import Session

from src.core.database.connector import ConnectorPostgres
from src.modules.diarization.infrastructure.repositories.models.diarization_task_model import (
    DiarizationTaskModel,
)

logger = logging.getLogger(__name__)
ACTIVE_STEPS = (
    "STARTED",
    "PROCESSING",
    "TRANSCRIPTION",
    "ALIGNMENT",
    "DIARIZATION",
    "DIARIZED",
)
LEASE_SECONDS = 120


class DiarizationTaskRepository:
    @staticmethod
    def _record_transition(
        db: Session,
        task: DiarizationTaskModel,
        previous_step: str | None,
        new_step: str,
        details: str | None = None,
    ) -> None:
        # Match SignalCatcher's identity: the task owns its history, even when
        # no external video is linked. Preserve the existing best-effort policy.
        db.flush()
        try:
            with db.begin_nested():
                db.execute(
                    text("""
                        INSERT INTO step_tracking
                            (entity_id, entity_type, previous_step, new_step, changed_at, details)
                        VALUES
                            (:entity_id, :entity_type, :prev_step, :new_step, :changed_at, :details)
                    """),
                    {
                        "entity_id": str(task.id),
                        "entity_type": "diarization",
                        "prev_step": previous_step,
                        "new_step": new_step,
                        "changed_at": datetime.now(timezone.utc).replace(tzinfo=None),
                        "details": details,
                    },
                )
        except Exception:
            logger.exception(
                "Failed to record diarization transition for task %s", task.id
            )

    def create_task(
        self,
        file_path: str,
        entity_id: str | None = None,
        entity_type: str | None = None,
        language: str | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        model_size: str = "large-v2",
    ) -> DiarizationTaskModel:
        with ConnectorPostgres() as db:
            if entity_id:
                if db.get_bind().dialect.name == "postgresql":
                    db.execute(
                        text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
                        {"key": f"diarization:{entity_type}:{entity_id}"},
                    )
                active = (
                    db.query(DiarizationTaskModel)
                    .filter(
                        DiarizationTaskModel.entity_id == entity_id,
                        DiarizationTaskModel.entity_type == entity_type,
                        DiarizationTaskModel.step.in_(("PENDING", *ACTIVE_STEPS)),
                    )
                    .order_by(
                        DiarizationTaskModel.created_at.desc(),
                        DiarizationTaskModel.id.desc(),
                    )
                    .first()
                )
                if active:
                    db.commit()
                    db.refresh(active)
                    db.expunge(active)
                    return active
            new_task = DiarizationTaskModel(
                file_path=file_path,
                step="PENDING",
                entity_id=entity_id,
                entity_type=entity_type,
                language=language,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                model_size=model_size,
            )
            db.add(new_task)
            self._record_transition(db, new_task, None, "PENDING")
            db.commit()
            db.refresh(new_task)
            # Create a detached copy to return
            db.expunge(new_task)
            return new_task

    def get_task(self, task_id: str) -> DiarizationTaskModel | None:
        with ConnectorPostgres() as db:
            task = (
                db.query(DiarizationTaskModel)
                .filter(DiarizationTaskModel.id == task_id)
                .first()
            )
            if task:
                db.expunge(task)
            return task

    def get_pending_tasks(self, limit: int = 5) -> list[DiarizationTaskModel]:
        with ConnectorPostgres() as db:
            tasks = (
                db.query(DiarizationTaskModel)
                .filter(DiarizationTaskModel.step == "PENDING")
                .order_by(DiarizationTaskModel.created_at.asc())
                .limit(limit)
                .all()
            )
            for task in tasks:
                db.expunge(task)
            return tasks

    def update_task_step(
        self,
        task_id: str,
        step: str,
        result_json: dict | None = None,
        error_message: str | None = None,
        worker_token: str | None = None,
    ) -> bool:
        with ConnectorPostgres() as db:
            task = (
                db.query(DiarizationTaskModel)
                .filter(DiarizationTaskModel.id == task_id)
                .with_for_update()
                .first()
            )
            if task:
                if worker_token is not None and (
                    task.worker_token != worker_token or task.step not in ACTIVE_STEPS
                ):
                    return False
                old_step = task.step
                task.step = step
                if old_step != step:
                    task.progress_percent = None
                if result_json is not None:
                    task.result_json = result_json
                if error_message is not None:
                    task.error_message = error_message
                if step in {"COMPLETED", "ERROR", "CANCELLED"}:
                    task.worker_token = None
                    task.lease_expires_at = None

                if old_step != step:
                    self._record_transition(db, task, old_step, step, error_message)

                db.commit()
                logger.info(f"Task {task_id} step updated to {step}")
                return True
            else:
                logger.warning(f"Task {task_id} not found for step update")
                return False

    def update_task_progress(
        self, task_id: str, step: str, percent: float, worker_token: str | None = None
    ) -> None:
        """Update telemetry without adding transitions or reviving cancelled tasks."""
        if step not in {"TRANSCRIPTION", "ALIGNMENT", "DIARIZATION"}:
            return
        value = max(0, min(100, int(percent)))
        with ConnectorPostgres() as db:
            query = db.query(DiarizationTaskModel).filter(
                DiarizationTaskModel.id == task_id,
                DiarizationTaskModel.step == step,
                (DiarizationTaskModel.progress_percent.is_(None))
                | (DiarizationTaskModel.progress_percent < value),
            )
            if worker_token is not None:
                query = query.filter(DiarizationTaskModel.worker_token == worker_token)
            query.update({DiarizationTaskModel.progress_percent: value})
            db.commit()

    def claim_task(self, task_id: str, token: str) -> bool:
        with ConnectorPostgres() as db:
            task = (
                db.query(DiarizationTaskModel)
                .filter(
                    DiarizationTaskModel.id == task_id,
                    DiarizationTaskModel.step == "PENDING",
                )
                .with_for_update(skip_locked=True)
                .first()
            )
            if task is None:
                return False
            task.worker_token = token
            task.lease_expires_at = datetime.now(timezone.utc).replace(
                tzinfo=None
            ) + timedelta(seconds=LEASE_SECONDS)
            task.progress_percent = None
            task.step = "TRANSCRIPTION"
            self._record_transition(db, task, "PENDING", "TRANSCRIPTION")
            db.commit()
            return True

    def renew_task_lease(self, task_id: str, token: str) -> bool:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with ConnectorPostgres() as db:
            count = (
                db.query(DiarizationTaskModel)
                .filter(
                    DiarizationTaskModel.id == task_id,
                    DiarizationTaskModel.worker_token == token,
                    DiarizationTaskModel.step.in_(ACTIVE_STEPS),
                    DiarizationTaskModel.lease_expires_at > now,
                )
                .update(
                    {
                        DiarizationTaskModel.lease_expires_at: now
                        + timedelta(seconds=LEASE_SECONDS)
                    }
                )
            )
            db.commit()
            return count == 1

    def recover_interrupted_tasks(self, include_legacy: bool = False) -> int:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with ConnectorPostgres() as db:
            expired = DiarizationTaskModel.lease_expires_at <= now
            if include_legacy:
                expired = or_(expired, DiarizationTaskModel.lease_expires_at.is_(None))
            tasks = (
                db.query(DiarizationTaskModel)
                .filter(
                    DiarizationTaskModel.step.in_(ACTIVE_STEPS),
                    expired,
                )
                .with_for_update(skip_locked=True)
                .all()
            )
            for task in tasks:
                previous = task.step
                task.step = "CANCELLED"
                task.progress_percent = None
                task.worker_token = None
                task.lease_expires_at = None
                task.error_message = (
                    "Processing interrupted: worker stopped renewing its lease."
                )
                self._record_transition(
                    db, task, previous, "CANCELLED", task.error_message
                )
            db.commit()
            return len(tasks)
