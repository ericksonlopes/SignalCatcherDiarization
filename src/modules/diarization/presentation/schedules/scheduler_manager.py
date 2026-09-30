import logging

from apscheduler.schedulers.background import BackgroundScheduler

from src.modules.diarization.infrastructure.repositories.diarization_task_repository import (
    DiarizationTaskRepository,
)
from src.modules.diarization.presentation.schedules.jobs.process_pending_diarization_job import (
    process_pending_diarization_tasks_job,
)

logger = logging.getLogger(__name__)


def start_scheduler() -> BackgroundScheduler:
    recovered = DiarizationTaskRepository().recover_interrupted_tasks(
        include_legacy=True
    )
    if recovered:
        logger.warning("Cancelled %s interrupted diarization tasks", recovered)
    scheduler = BackgroundScheduler()

    scheduler.add_job(
        process_pending_diarization_tasks_job,
        trigger="interval",
        seconds=10,
        id="process_diarization_tasks",
        replace_existing=True,
        max_instances=1,
    )

    logger.info("Scheduler de diarização iniciado em background.")
    scheduler.start()

    return scheduler
