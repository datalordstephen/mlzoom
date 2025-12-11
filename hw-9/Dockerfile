FROM agrigorev/model-2025-hairstyle:v1

RUN pip install onnxruntime pillow numpy

COPY lambda_function.py ./

CMD ["lambda_function.lambda_handler"]


# # get python and uv
# FROM public.ecr.aws/lambda/python:3.13
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

# # intall deps with uv (globally not locally)
# COPY pyproject.toml uv.lock ./
# RUN uv pip install --system -r <(uv export --format requirements-txt)

# # copy function and model
# COPY lambda_function.py model.bin ./

# # invoke lambda func
# CMD ["lambda_function.lambda_handler"]

