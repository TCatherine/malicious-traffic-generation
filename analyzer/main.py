from jinja2 import Environment, FileSystemLoader
from datetime import datetime
from tqdm import tqdm

s = 'admin/mediaAdmin.php?Id=%22%3E%3Ciframe%20src=a%20onload=alert%28%22VL%22%29%20%3C'
example = [
    "boafrm/=%3E%3Cscript%3E",
    "admin/mediaAdmin.php?id=%22%3E%3Ciframe%20src=a%20onload=alert%28%22VL%22%29%20%3C",
    "search.php?test=query$=document,$=$.URL,$$=unescape,$$$=eval,$$$($$($))%0",
    "index.php?test=queryevil=/ev/.source+/al/.source,changeProto=/Strin/.source+/g.prototyp/.source+/e.ss=/.source+/Strin/.source+/g.prototyp/.source+/e.substrin/.source+/g/.source,hshCod=/documen/.source+/t.locatio/.source+/n.has/.source+/h/.source;7%5Bevil%5D(changeProto);hsh=7%5Bevil%5D(hshCod),cod=hsh.ss(1);7%5Bevil%5D(cod)%0",
    "index.html?test=query%3CSTYLE%20TYPE=%22text/javascript%22%3Ealert('XSS');%3C/STYLE%3E%0",
    "search.php?test=queryA=alert;A(1)%0"
]

environment = Environment(loader=FileSystemLoader("./"))
template = environment.get_template("rule.j2")

for s in tqdm(example):
    rev = 1
    sid = abs(hash(s)) % (10 ** 8)
    data_create = datetime.today().strftime('%Y_%m_%d')

    content = template.render(
            content=s,
            rev=rev,
            sid=sid,
            data=data_create
        )

    with open("../SEIM/results/unidentified_packets", 'a') as f:
        f.write(f"{s}\n")

    with open("../SEIM/results/rules", 'a') as f:
        f.write(f"{content}\n")
