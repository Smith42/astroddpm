#!/bin/bash

NAME=$(echo $1 | cut -d',' -f1)
RA=$(echo $1 | cut -d',' -f2)
DEC=$(echo $1 | cut -d',' -f3)
Z=$(echo $1 | cut -d',' -f10)
echo Dumping $NAME $RA $DEC $Z
wget -nc -O raws/${NAME}_g.fits "http://legacysurvey.org/viewer/fits-cutout?ra=$RA&dec=$DEC&size=512&layer=ls-dr9&pixscale=0.262&bands=g"
wget -nc -O raws/${NAME}_r.fits "http://legacysurvey.org/viewer/fits-cutout?ra=$RA&dec=$DEC&size=512&layer=ls-dr9&pixscale=0.262&bands=r"
wget -nc -O raws/${NAME}_z.fits "http://legacysurvey.org/viewer/fits-cutout?ra=$RA&dec=$DEC&size=512&layer=ls-dr9&pixscale=0.262&bands=z"
echo $NAME >> done.txt
