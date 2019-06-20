#!/bin/bash

set -e
if [ ! -d ../src ]; then
    mkdir ../src/
fi
cd ../src/

# Install speech_dtw
if [ ! -d speech_dtw ]; then
    git clone https://github.com/kamperh/speech_dtw.git
    cd speech_dtw
    make
    make test
    cd -
fi

# Install shorten
if [ ! -d shorten-3.6.1 ]; then
    wget https://download.tuxfamily.org/xcfaudio/PROG_ABS_FRUGALWARE/SHORTEN/shorten-3.6.1.tar.gz
    # wget http://etree.org/shnutils/shorten/dist/src/shorten-3.6.1.tar.gz
    tar -zxf shorten-3.6.1.tar.gz
    cd shorten-3.6.1
    ./configure --prefix=`pwd`
    make
    make install
fi

set +e
