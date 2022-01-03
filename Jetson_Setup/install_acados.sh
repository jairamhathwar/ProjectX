#!/bin/bash

git clone --recursive https://github.com/acados/acados.git
cd acados
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_HPMPC=OFF -DACADOS_WITH_QORE=ON -DACADOS_WITH_OOQP=ON ACADOS_WITH_QPDUNES=ON -DACADOS_WITH_OSQP=ON ..

