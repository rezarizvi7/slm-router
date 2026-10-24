FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY slm_router ./slm_router
RUN pip install --no-cache-dir .
EXPOSE 8080
CMD ["slm-router"]
