# image to pull
FROM python:3.12.1-slim-bookworm

# get package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# set working directory
WORKDIR /app

# add venv to path so that installed packages are found globally
ENV PATH="/app/.venv/bin:$PATH"

# copy useful files to working directory
COPY [".python-version", "pyproject.toml", "uv.lock", "./"]

# install dependencies (exact versions)
RUN uv sync --locked

# copy useful files to pwd
COPY ["app.py", "model.bin", "./"]

# expose port
EXPOSE 9696

# set entrypoint
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]

