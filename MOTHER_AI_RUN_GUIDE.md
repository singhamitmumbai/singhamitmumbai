# Mother AI — Run Guide

```bash
# Integrated Mother AI (default)
python -m cognitive_ai
# or
python -m cognitive_ai mother

Mother> hii
Mother> how are you
Mother> inspect this project and find what is broken
Mother> Show unpaid invoices for Project A
Mother> status
Mother> diagnose
Mother> benchmark
Mother> quit

# One-shots
python -m cognitive_ai status
python -m cognitive_ai diagnose
python -m cognitive_ai benchmark
python -m cognitive_ai mother --ask "kaise ho"

# Classic symbolic commands still work inside Mother>
Mother> domain animals
Mother> assert tweety is_a canary
Mother> think 3
```

Reports live under `reports/mother_ai/` and the root `MOTHER_AI_*.md` files.
