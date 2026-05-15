# python=3.12

```bash
git clone https://github.com/nguyenduchuyiu/BetaZero.git
cd BetaZero
cd repl
lake update 
lake build
git lfs install
git lfs pull
pip install -r requirements.txt
python betazero/train.py --config configs/deepseek_r1_distill_qwen_7B.yaml
```

```bash
tensorboard --logdir outputs/runs/ serve
```
