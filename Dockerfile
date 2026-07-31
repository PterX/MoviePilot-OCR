FROM python:3.12-slim-bookworm

ENV LANG=C.UTF-8 \
    TZ=Asia/Shanghai \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py ./

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 9899

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9899/', timeout=3)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9899"]
