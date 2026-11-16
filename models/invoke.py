
from init_llm import deep_seek_llm


resp = deep_seek_llm.invoke("请简单说明一下")

print(resp.content)