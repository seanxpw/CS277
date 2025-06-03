FROM python:3.10-bookworm AS builder

ENV NNI_RELEASE=3.0.1

RUN git clone https://github.com/KashunCheng/nni.git && \
    cd nni && \
    pip install jupyterlab==3.0.9 && \
    python setup.py build_ts && \
    python setup.py bdist_wheel

FROM ghcr.io/astral-sh/uv:python3.10-bookworm

WORKDIR /app/

COPY --from=builder /nni/dist/nni-3.0.1-py3-none-any.whl /nni-3.0.1-py3-none-any.whl
COPY ./uv.lock .
COPY ./pyproject.toml .

RUN uv sync --locked

ENTRYPOINT ["tail", "-f", "/dev/null"]