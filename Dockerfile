FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# leads.db and checkpoints.sqlite must exist as files (not directories)
# before a volume gets mounted over them, or Docker creates an empty
# directory at that path instead and sqlite3.connect() breaks. Pre-creating
# them here means `docker run -v` (see README) mounts cleanly either way.
RUN touch leads.db checkpoints.sqlite && mkdir -p index

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
