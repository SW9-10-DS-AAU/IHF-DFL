import pickle, sys, glob
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
f = sys.argv[1]
p = pickle.load(open(f, "rb"))
print(p.keys())
print("META", p["metadata"])
print("SETUP", p.get("setup"))
for k, v in p["tables"].items():
    print("==", k, getattr(v, "shape", None))
    if hasattr(v, "columns"):
        print(list(v.columns))
        print(v.head(8).to_string()[:3000])
