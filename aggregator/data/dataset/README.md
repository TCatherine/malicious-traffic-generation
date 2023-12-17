# Reference model

Collection of reference samples

## XSS Injection

Утилита: https://www.geeksforgeeks.org/xss-loader-xss-scanner-and-payload-generator/

Запуск:

```bash
python3 payloader.py
```

Захват трафика:

```bash
sudo tcpdump -i any -A -s 10240 'tcp port 80 and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0)' | egrep --line-buffered "^........(GET |HTTP\/|POST |HEAD )|^[A-Za-z0-9-]+: " | sed -r 's/^........(GET |HTTP\/|POST |HEAD )/\n\1/g' > xss
```
