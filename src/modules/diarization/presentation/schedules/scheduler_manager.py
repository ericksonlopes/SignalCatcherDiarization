import logging

from apscheduler.schedulers.background import BackgroundScheduler

from src.modules.diarization.presentation.schedules.jobs.process_pending_diarization_job import \
    process_pending_diarization_tasks_job

logger = logging.getLogger(__name__)


def start_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler()

    scheduler.add_job(
        process_pending_diarization_tasks_job,
        trigger="interval",
        seconds=10,
        id="process_diarization_tasks",
        replace_existing=True,
        max_instances=1
    )

    logger.info("Scheduler de diarização iniciado em background.")
    scheduler.start()

    return scheduler
