WEIGHTS_FILE_ID=1uSB20w3ioWB8BHS5VGeX_C8f_TrMcgpE
gdown https://drive.google.com/uc?id=${WEIGHTS_FILE_ID}
unzip -n weights.zip -d .
rm weights.zip