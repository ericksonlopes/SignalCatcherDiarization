import logging
from typing import Optional, List
from sqlalchemy.orm import Session
from src.modules.diarization.infrastructure.repositories.models.diarization_task_model import DiarizationTaskModel
from src.core.database.connector import ConnectorPostgres

logger = logging.getLogger(__name__)

class DiarizationTaskRepository:
    def create_task(
        self,
        file_path: str,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        language: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        model_size: str = "large-v2"
    ) -> DiarizationTaskModel:
        with ConnectorPostgres() as db:
            new_task = DiarizationTaskModel(
                file_path=file_path,
                step="PENDING",
                entity_id=entity_id,
                entity_type=entity_type,
                language=language,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                model_size=model_size
            )
            db.add(new_task)
            db.commit()
            db.refresh(new_task)
            # Create a detached copy to return
            db.expunge(new_task)
            return new_task

    def get_task(self, task_id: str) -> Optional[DiarizationTaskModel]:
        with ConnectorPostgres() as db:
            task = db.query(DiarizationTaskModel).filter(DiarizationTaskModel.id == task_id).first()
            if task:
                db.expunge(task)
            return task

    def get_pending_tasks(self, limit: int = 5) -> List[DiarizationTaskModel]:
        with ConnectorPostgres() as db:
            tasks = db.query(DiarizationTaskModel).filter(
                DiarizationTaskModel.step == "PENDING"
            ).order_by(DiarizationTaskModel.created_at.asc()).limit(limit).all()
            for task in tasks:
                db.expunge(task)
            return tasks

    def update_task_step(self, task_id: str, step: str, result_json: Optional[dict] = None, error_message: Optional[str] = None) -> None:
        from sqlalchemy import text
        
        with ConnectorPostgres() as db:
            task = db.query(DiarizationTaskModel).filter(DiarizationTaskModel.id == task_id).first()
            if task:
                old_step = task.step
                task.step = step
                if result_json is not None:
                    task.result_json = result_json
                if error_message is not None:
                    task.error_message = error_message
                    
                # Insert into step_tracking if entity_id exists
                if task.entity_id and task.entity_type and old_step != step:
                    try:
                        # We use text() to insert raw SQL because we don't have StepTrackingModel imported here
                        db.execute(
                            text('''
                                INSERT INTO step_tracking (entity_id, entity_type, previous_step, new_step, changed_at, details)
                                VALUES (:entity_id, :entity_type, :prev_step, :new_step, NOW(), :details)
                            '''),
                            {
                                "entity_id": task.entity_id,
                                "entity_type": task.entity_type,
                                "prev_step": old_step,
                                "new_step": step,
                                "details": error_message
                            }
                        )
                    except Exception as e:
                        logger.error(f"Failed to insert into step_tracking: {e}")

                db.commit()
                logger.info(f"Task {task_id} step updated to {step}")
            else:
                logger.warning(f"Task {task_id} not found for step update")
