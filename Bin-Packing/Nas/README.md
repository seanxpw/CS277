## Build
```bash
jchen1192@onix:~/projects/CS277$ pwd
/nfshome/jchen1192/projects/CS277
jchen1192@onix:~/projects/CS277$ docker build -t gc -f ./GraphCover/Nas/Dockerfile .
```
## Run and attach
```bash
docker run --gpus all -it -d --rm -v `pwd`:/app --name graph-cover gc 
docker exec -it graph-cover /bin/bash 
```

## Train
```bash
uv venv
uv sync
. /app/.venv/bin/activate # or . .venv/bin/activate
uv add /nni-3.0.1-py3-none-any.whl
python model.py --data-set ../Original/vertex_cover_btute_force_20250601_183711/
```