FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GITCONTEXT_INDEX_DIR=/data/index

# git is required by the repository connector (git log / ls-files)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[anthropic,openai]"
COPY examples ./examples
COPY evals ./evals

RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown -R app:app /data /app
USER app

EXPOSE 8000
# Index the bundled sample repo on start-up, then serve. Mount your own repo and override CMD:
#   docker run -v $PWD:/repo gitcontext sh -c "ask-repo index /repo && ask-repo serve --host 0.0.0.0"
CMD ["sh", "-c", "ask-repo sample-repo /data/shopfront-api && ask-repo index /data/shopfront-api && ask-repo serve --host 0.0.0.0 --port 8000"]
