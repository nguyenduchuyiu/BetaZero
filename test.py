import json
# p="automation-output/verified_failed.json"
# d=json.load(open(p,"r",encoding="utf-8"))
# rows=[(x.get("id","<no-id>"),x.get("error_reason","")) for x in d if "Gate Failed" in (x.get("error_reason","") or "")]
# for i,(sid,err) in enumerate(rows,1):
#     print(f"{i:03d}. {sid} | {err}")
# print("---\nTOTAL:",len(rows))

p="automation-output/SFT_data.json"
d=json.load(open(p,"r",encoding="utf-8"))
print("---\nTOTAL:",len(d))