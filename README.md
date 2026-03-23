# Install
0. Clone
```bash
git clone https://github.com/nguyenduchuyiu/BetaZero
cd BetaZero
```
1. Install lake
```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```
2. Build 
```bash
cd repl
lake exe cache get
# if above command fails, build directly
lake build
```
3. Run Auto Sorrifier
First time running can be extremely slow due to building Matlib from cache.
```bash
python sorrifier_usage_example.py
```