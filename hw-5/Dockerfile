FROM agrigorev/zoomcamp-model:2025

# get package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# set working directory
WORKDIR /code

# add venv to path so that installed packages are found globally
ENV PATH="/code/.venv/bin:$PATH"

# copy useful files to working directory
COPY [".python-version", "pyproject.toml", "uv.lock", "./"]

# install dependencies (exact versions)
RUN uv sync --locked

# copy useful files to pwd
COPY ["app.py", "./"]

# expose port
EXPOSE 9696

# set entrypoint
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]