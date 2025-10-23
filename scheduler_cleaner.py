# scheduler_cleaner.py
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import atexit
from nettoyage_tables import clean_old_records_multi  # ta fonction actuelle

def start_scheduler():
    scheduler = BackgroundScheduler(timezone="UTC")

    # tâche quotidienne à 03h00 UTC
    scheduler.add_job(
        clean_old_records_multi,
        trigger='cron',
        hour=3, minute=0,
        id='clean_old_records',
        replace_existing=True
    )

    scheduler.start()
    atexit.register(lambda: scheduler.shutdown(wait=False))
    print(f"[✅] APScheduler lancé à {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC (nettoyage quotidien à 03h00)")
