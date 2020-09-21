#!/usr/bin/env bash

V=$(date -u +%Y.%m.%d);

./wheel_info.py requirements \
    ../dist/narchi-*-py3-none-any.whl \
    --extras_require pytorch \
  | grep -v torch \
  > requirements.txt;

docker build -t mauvilsa/narchi-test:$V-py38 .;

for py in 3.6 3.7; do
  echo "FROM mauvilsa/narchi-test:$V-py38
RUN ln -fs python$py /usr/bin/python3 && ln -fs pip$py /usr/local/bin/pip3
  " > py$py.Docker;
  docker build -t mauvilsa/narchi-test:$V-py${py/./} -f py$py.Docker .;
  rm py$py.Docker;
done
