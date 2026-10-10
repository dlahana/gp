FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY tennispred ./tennispred
# Keep the SQLite DB and fitted reward model on a mounted volume.
ENV BOT_DB_PATH=/data/bot.db BOT_REWARD_PATH=/data/reward.json
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "tennispred.server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
