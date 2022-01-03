#!/bin/bash

# need to fix the render 
# https://discourse.acados.org/t/problems-with-t-renderer/438

# install rust
curl https://sh.rustup.rs -sSf -o install_rust.sh
sh install_rust.sh -y
rm install_rust.sh

# get tera_renderer
git clone https://github.com/acados/tera_renderer.git
cd tera_renderer
cargo build --verbose --release
cd ..

git clone --recursive https://github.com/acados/acados.git
cd acados
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_HPMPC=OFF -DACADOS_WITH_QORE=ON -DACADOS_WITH_OOQP=ON ACADOS_WITH_QPDUNES=ON -DACADOS_WITH_OSQP=ON ..

# copy the tera_renderer
cd ..
cp ../tera_renderer/target/release/t_renderer bin/
