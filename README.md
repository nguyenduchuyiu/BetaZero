# Setup Environment

```bash
git clone https://github.com/nguyenduchuyiu/BetaZero.git
cd BetaZero
git lfs install
git lfs pull
pip install -r requirements.txt
cd repl 
lake update
lake build 
lake build dump_ast_server
lake build dump_expr_server
```

```bash
tensorboard --logdir outputs/runs/ serve
```

# Run visualization
```bash
./serve.sh
```
Open http://localhost:1234/and_or_graph.html in browser
Choose json file fron outputs/rollouts/gemini3flash