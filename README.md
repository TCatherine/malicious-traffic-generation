## Environment
```
python3.11 -m venv gen
```

## Входные данные
### Генерация XSS
URL: https://www.geeksforgeeks.org/xss-loader-xss-scanner-and-payload-generator/
```bash
python3 payloader.py
```

Захват трафика:
```bash
sudo tcpdump -i any -A -s 10240 'tcp port 80 and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0)' | egrep --line-buffered "^........(GET |HTTP\/|POST |HEAD )|^[A-Za-z0-9-]+: " | sed -r 's/^........(GET |HTTP\/|POST |HEAD )/\n\1/g' > xss
```

## Byte pair encoding
URL: https://github.com/soaxelbrooke/python-bpe
```bash
git submodule init
git submodule update
pip install -r parser/bpe/requirements.txt
```
