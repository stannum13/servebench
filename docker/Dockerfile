FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY README.md ./
COPY REPORT.md BENCH_STATE.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY configs ./configs
RUN mkdir -p /app/results /app/figures
CMD ["uvicorn", "router.app:app", "--host", "0.0.0.0", "--port", "8080"]
